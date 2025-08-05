/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorchexamples.dl3;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.Build;
import android.os.Environment;
import android.os.SystemClock;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import org.pytorch.executorch.EValue;
import org.pytorch.executorch.Module;
import org.pytorch.executorch.Tensor;

import java.io.InputStream;
import java.io.FileOutputStream;
import java.io.File;
import java.io.IOException;

import android.net.Uri;
import android.content.Intent;
import android.provider.MediaStore;
import androidx.core.content.FileProvider;

import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.File;
import java.io.IOException;


public class MainActivity extends Activity implements Runnable {
  private ImageView mImageView;
  private Button mButtonXnnpack;
  private ProgressBar mProgressBar;
  private Bitmap mBitmap = null;
  private Module mModule = null;
  private String mImagename = "corgi.jpeg";

  private final ArrayList<String> mImageFiles = new ArrayList<>();

  private int mCurrentImageIndex = 0;

  private static final int REQUEST_READ_EXTERNAL_STORAGE = 1001;
  private static final String LOCAL_IMAGE_DIR = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES).getAbsolutePath() + "/";

  // see http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2007/segexamples/index.html for the list of
  // classes with indexes
  private static final int CLASSNUM = 21;
  private static final int DOG = 12;
  private static final int PERSON = 15;
  private static final int SHEEP = 17;
  private static final int REQUEST_IMAGE_CAPTURE = 2001;
  private Uri mPhotoUri;
  private File mCapturedPhotoFile;

  private static final int REQUEST_CAMERA_PERMISSION = 3001;
  private void checkCameraPermissionAndLaunch() {
    if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
      ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA_PERMISSION);
    } else {
      dispatchTakePictureIntent();
    }
  }

  private void checkAndRequestStoragePermission() {
    if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE)
            != PackageManager.PERMISSION_GRANTED) {
      // Permission is not granted, request it
      if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) { // Android 13+
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_MEDIA_IMAGES)
                != PackageManager.PERMISSION_GRANTED) {
          ActivityCompat.requestPermissions(this,
                  new String[]{Manifest.permission.READ_MEDIA_IMAGES},
                  REQUEST_READ_EXTERNAL_STORAGE);
        } else {
          // Permission already granted, proceed with file access
          loadImagesFromLocal();
        }
      } else {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED) {
          ActivityCompat.requestPermissions(this,
                  new String[]{Manifest.permission.READ_EXTERNAL_STORAGE},
                  REQUEST_READ_EXTERNAL_STORAGE);
        } else {
          // Permission already granted, proceed with file access
          loadImagesFromLocal();
        }
      }
    } else {
      // Permission already granted, proceed with file access
      loadImagesFromLocal();
    }
  }

  // Handle the permission request result
  @Override
  public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    if (requestCode == REQUEST_CAMERA_PERMISSION) {
      if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
        dispatchTakePictureIntent();
      } else {
        Toast.makeText(this, "Camera permission is required", Toast.LENGTH_SHORT).show();
      }
    }
    if (requestCode == REQUEST_READ_EXTERNAL_STORAGE) {
      if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
        // Permission granted
        loadImagesFromLocal();
      } else {
        // Permission denied, show a message or fallback
        showUIMessage(this, "Permission denied to read external storage");
      }
    }

    // Load images from assets anyways
    populateImagePathFromAssets();
  }

  private void loadImagesFromLocal() {
    // Load images from /data/local & /sdcard/Pictures
    boolean hasAnyLocalFiles = populateImagePathFromLocal();
    boolean isImageShown = showImage();
    if (hasAnyLocalFiles && isImageShown) {
      showUIMessage(this, "Refreshed images from external storage and/or Assets folder");
    }
  }

  private void dispatchTakePictureIntent() {
    Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
    if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
      try {
        mCapturedPhotoFile = createImageFile();
        mPhotoUri = FileProvider.getUriForFile(this,
                getApplicationContext().getPackageName() + ".provider", mCapturedPhotoFile);
        takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, mPhotoUri);
        startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
      } catch (IOException ex) {
        Log.e("Camera", "Error creating image file", ex);
      }
    }
  }

  @Override
  protected void onActivityResult(int requestCode, int resultCode, Intent data) {
    super.onActivityResult(requestCode, resultCode, data);
    if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
      mImagename = mCapturedPhotoFile.getAbsolutePath();
      mBitmap = BitmapFactory.decodeFile(mImagename);
      if (mBitmap != null) {
        mBitmap = Bitmap.createScaledBitmap(mBitmap, 224, 224, true);
        mImageView.setImageBitmap(mBitmap);

        // Run segmentation thread
        mProgressBar.setVisibility(ProgressBar.VISIBLE);
        Thread thread = new Thread(MainActivity.this);
        thread.start();
      } else {
        Toast.makeText(this, "Failed to load captured image", Toast.LENGTH_SHORT).show();
      }
    }
  }

  private File createImageFile() throws IOException {
    String fileName = "captured_photo_" + System.currentTimeMillis();
    File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
    File image = File.createTempFile(fileName, ".jpg", storageDir);
    return image;
  }

  private boolean showImage() {
    boolean isImageShown = false;
    if (mImagename == null) {
      showUIMessage(this, "No image to display");
      mImageView.setImageBitmap(null);
      return isImageShown;
    }
    try {
      if (mImagename.startsWith("/")) {
        mBitmap = BitmapFactory.decodeFile(mImagename);
      } else {
        mBitmap = BitmapFactory.decodeStream(getAssets().open(mImagename));
      }
      if (mBitmap != null) {
        mBitmap = Bitmap.createScaledBitmap(mBitmap, 224, 224, true);
        mImageView.setImageBitmap(mBitmap);
        isImageShown = true;
      }
    } catch (IOException e) {
      Log.e("ImageSegmentation", "Error reading image", e);
      mImageView.setImageBitmap(null);
    }
    return isImageShown;
  }

  private boolean populateImagePathFromLocal() {
    boolean hasLocalFiles = false;
    File dir = new File(LOCAL_IMAGE_DIR);
    File[] files = dir.listFiles((d, name) ->
            name.endsWith(".jpg") || name.endsWith(".jpeg") || name.endsWith(".png"));
    ArrayList<String> imageList = new ArrayList<>();
    if (files != null && files.length > 0) {
      for (int i = 0; i < files.length; i++) {
        mImageFiles.add(files[i].getAbsolutePath());
        hasLocalFiles = true;
      }
      mImagename = mImageFiles.get(0);
    } else {
      mImagename = null;
    }

    return hasLocalFiles;
  }

  private void populateImagePathFromAssets() {
    try {
      String[] allFiles = getAssets().list("");
      if (allFiles != null && allFiles.length > 0) {
        for (String file : allFiles) {
          if (file.endsWith(".jpg") || file.endsWith(".jpeg") || file.endsWith(".png")) {
            mImageFiles.add(file);
          }
        }
        mCurrentImageIndex = 0;
        mImagename = !mImageFiles.isEmpty() ? mImageFiles.get(0) : null;
      }
    } catch (IOException e) {
      Log.e("ImageSegmentation", "Error listing assets", e);
      finish();
    }
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    // Initialize all views first!
    mImageView = findViewById(R.id.imageView);
    mButtonXnnpack = findViewById(R.id.xnnpackButton);
    mProgressBar = findViewById(R.id.progressBar);

    populateImagePathFromAssets();
    showImage();

    try {
      String modelPath = copyAssetToFile(this, "dl3_xnnpack_fp32.pte");
      mModule = Module.load(modelPath);
    } catch (IOException e) {
      Log.e("ModelLoad", "Failed to load model from assets", e);
      showUIMessage(this, "Failed to load model from assets");
      return;
    }

    final Button capturePhotoButton = findViewById(R.id.capturePhotoButton);
    capturePhotoButton.setOnClickListener(v -> checkCameraPermissionAndLaunch());




//    mModule = Module.load("/data/local/tmp/dl3_xnnpack_fp32.pte");
    mImageView.setImageBitmap(mBitmap);

    final Button buttonNext = findViewById(R.id.nextButton);
    buttonNext.setOnClickListener(
        new View.OnClickListener() {
          public void onClick(View v) {
            if (mImageFiles == null || mImageFiles.isEmpty()) {
              // No images available
              return;
            }
            // Move to the next image, wrap around if at the end
            mCurrentImageIndex = (mCurrentImageIndex + 1) % mImageFiles.size();
            mImagename = mImageFiles.get(mCurrentImageIndex);
            showImage();
          }
        });

    mButtonXnnpack.setOnClickListener(
        new View.OnClickListener() {
          public void onClick(View v) {
            mModule.destroy();
//            mModule = Module.load("/data/local/tmp/dl3_xnnpack_fp32.pte");
            try {
              mModule.destroy();
              String modelPath = copyAssetToFile(MainActivity.this, "dl3_xnnpack_fp32.pte");
              mModule = Module.load(modelPath);
            } catch (IOException e) {
              Log.e("ModelReload", "Failed to reload model from assets", e);
              showUIMessage(MainActivity.this, "Model reload failed");
              return;
            }

            mButtonXnnpack.setEnabled(false);
            mProgressBar.setVisibility(ProgressBar.VISIBLE);
            mButtonXnnpack.setText(getString(R.string.run_model));

            Thread thread = new Thread(MainActivity.this);
            thread.start();
          }
        });

    final Button resetImage = findViewById(R.id.resetImage);
    resetImage.setOnClickListener(
            v -> showImage());

    // Refresh Button for External Storage
    final Button loadAndRefreshButton = findViewById(R.id.loadAndRefreshButton);
    loadAndRefreshButton.setOnClickListener(
        v -> {
          mImageFiles.clear();
          checkAndRequestStoragePermission();

          populateImagePathFromAssets();
          showImage();
        }
    );
  }

  @Override
  public void run() {
    final Tensor inputTensor =
        TensorImageUtils.bitmapToFloat32Tensor(
            mBitmap,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
            TensorImageUtils.TORCHVISION_NORM_STD_RGB);

    boolean imageSegementationSuccess = false;
    final long startTime = SystemClock.elapsedRealtime();
    Tensor outputTensor = mModule.forward(EValue.from(inputTensor))[0].toTensor();
    final long inferenceTime = SystemClock.elapsedRealtime() - startTime;
    Log.d("ImageSegmentation", "inference time (ms): " + inferenceTime);

    final float[] scores = outputTensor.getDataAsFloatArray();
    int width = mBitmap.getWidth();
    int height = mBitmap.getHeight();

    int[] intValues = new int[width * height];
    for (int j = 0; j < height; j++) {
      for (int k = 0; k < width; k++) {
        int maxi = 0, maxj = 0, maxk = 0;
        double maxnum = -Double.MAX_VALUE;
        for (int i = 0; i < CLASSNUM; i++) {
          float score = scores[i * (width * height) + j * width + k];
          if (score > maxnum) {
            maxnum = score;
            maxi = i;
            maxj = j;
            maxk = k;
          }
        }
        if (maxi == PERSON) intValues[maxj * width + maxk] = 0xFFFF0000; // R
        else if (maxi == DOG) intValues[maxj * width + maxk] = 0xFF00FF00; // G
        else if (maxi == SHEEP) intValues[maxj * width + maxk] = 0xFF0000FF; // B
        else intValues[maxj * width + maxk] = 0xFF000000;
        if (maxi == PERSON || maxi == DOG || maxi == SHEEP) {
          imageSegementationSuccess = true;
        }
      }
    }

    Bitmap bmpSegmentation = Bitmap.createScaledBitmap(mBitmap, width, height, true);
    Bitmap outputBitmap = bmpSegmentation.copy(bmpSegmentation.getConfig(), true);
    outputBitmap.setPixels(
        intValues,
        0,
        outputBitmap.getWidth(),
        0,
        0,
        outputBitmap.getWidth(),
        outputBitmap.getHeight());
    final Bitmap transferredBitmap =
        Bitmap.createScaledBitmap(outputBitmap, mBitmap.getWidth(), mBitmap.getHeight(), true);

    final boolean showUserIndicationOnImgSegFail = !imageSegementationSuccess;
    runOnUiThread(
            () -> {
              if (showUserIndicationOnImgSegFail) {
                Toast.makeText(this, "ImageSegmentation Failed", Toast.LENGTH_SHORT).show();
              }
              mImageView.setImageBitmap(transferredBitmap);
              mButtonXnnpack.setEnabled(true);
              mButtonXnnpack.setText(R.string.run_xnnpack);
              mProgressBar.setVisibility(ProgressBar.INVISIBLE);
            });
  }

  void showUIMessage(final Context context, final String msg) {
    runOnUiThread(new Runnable() {
      public void run() {
        Toast.makeText(context, msg, Toast.LENGTH_SHORT).show();
      }
    });
  }

  private String copyAssetToFile(Context context, String assetName) throws IOException {
    File file = new File(context.getFilesDir(), assetName);
    if (!file.exists()) {
      try (InputStream is = context.getAssets().open(assetName);
           FileOutputStream fos = new FileOutputStream(file)) {
        byte[] buffer = new byte[1024];
        int length;
        while ((length = is.read(buffer)) > 0) {
          fos.write(buffer, 0, length);
        }
      }
    }
    return file.getAbsolutePath();
  }

}
