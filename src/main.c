#include <math.h>
#include <raylib.h>
#include <raymath.h>
#include <raymath.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define NN_IMPLEMENTATION
#include "../thirdparty/nn.h"
#define CUBE_SIZE 20
#include "./parsers/vox_parser.c"

void draw_voxels(Voxel *model, Vector3 pos) {
  const float t = 0.05;
  for (size_t i = 0; i < CUBE_SIZE; i++) {
    for (size_t j = 0; j < CUBE_SIZE; j++) {
      for (size_t k = 0; k < CUBE_SIZE; k++) {
        if (model[index3D(j, i, k)].colorIndex != -1) {
          Vector3 lpos = Vector3Add((Vector3){j, k, i}, pos);
          DrawCubeV(lpos, (Vector3){1, 1, 1}, BLACK);
          DrawCubeWiresV((Vector3){lpos.x + t / 2.0, lpos.y + t / 2.0, lpos.z + t / 2.0},
                         (Vector3){1 + t, 1 + t, 1 + t}, WHITE);
        }
      }
    }
  }
}

int main(void) {
  srand(time(0));

  Layer layers[] = {
      (Layer){.size = 4},
      (Layer){.size = 7, .actf = ACT_RELU, .randf = glorot_randf},
      (Layer){.size = 7, .actf = ACT_RELU, .randf = glorot_randf},
      (Layer){.size = 7, .actf = ACT_RELU, .randf = glorot_randf},
      (Layer){.size = 7, .actf = ACT_RELU, .randf = glorot_randf},
      (Layer){.size = 1, .actf = ACT_SIGM, .randf = glorot_randf}
  };

  NN nn = nn_alloc(layers, ARR_LEN(layers));
  NN g = nn_alloc(layers, ARR_LEN(layers));

  nn_rand(nn);

  Voxel *torus = load_vox_model("models/torus.vox");
  Voxel *apple = load_vox_model("models/apple.vox");

  Mat input = mat_alloc(1, 4);
  Mat output = mat_alloc(1, 1);

  printf("Learning process started\n");

  // learning process
  const float learning_rate = 0.004;
  const int epochs = 3000;
  
  for (int epoch = 0; epoch < epochs; epoch++) {
    for (size_t i = 0; i < CUBE_SIZE; i++) {
      for (size_t j = 0; j < CUBE_SIZE; j++) {
        for (size_t k = 0; k < CUBE_SIZE; k++) {
          MAT_AT(input, 0, 0) = i / (CUBE_SIZE - 1.0);
          MAT_AT(input, 0, 1) = j / (CUBE_SIZE - 1.0);
          MAT_AT(input, 0, 2) = k / (CUBE_SIZE - 1.0);
          MAT_AT(input, 0, 3) = 0;
          MAT_AT(output, 0, 0) =
              (float)(torus[index3D(j, i, k)].colorIndex != -1);
          nn_backprop(nn, g, input, output);
          nn_learn(nn, g, learning_rate);

          MAT_AT(input, 0, 0) = i / (CUBE_SIZE - 1.0);
          MAT_AT(input, 0, 1) = j / (CUBE_SIZE - 1.0);
          MAT_AT(input, 0, 2) = k / (CUBE_SIZE - 1.0);
          MAT_AT(input, 0, 3) = 1;
          MAT_AT(output, 0, 0) =
              (float)(apple[index3D(j, i, k)].colorIndex != -1);
          nn_backprop(nn, g, input, output);
          nn_learn(nn, g, learning_rate);
        }
      }
    }

    if (epoch % 100 == 0)
      printf("epoch - %d\n", epoch);
  }

  printf("Learning ended. Use WASD to rotate the camera. Press ENTER to start visualization...");
  getchar(); 

  const int screenWidth = 800;
  const int screenHeight = 600;
  InitWindow(screenWidth, screenHeight, "Interp 3D");

  const float cameraDistance = 25;
  Vector3 target = {CUBE_SIZE/2.0, CUBE_SIZE/2.0, CUBE_SIZE/2.0};

  Camera camera = {0};
  camera.target = target;
  camera.up = (Vector3){0.0f, 1.0f, 0.0f};
  camera.fovy = 70.0f;
  camera.projection = CAMERA_PERSPECTIVE;

  float slider = 0.0;
  float c_x = 0.0, c_y = 0.0;

  while (!WindowShouldClose()) {
    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
      slider = fmin(fmax(GetMouseX() / (float)screenWidth, 0.0), 1.0);
    }

    if (IsKeyDown(KEY_A)) c_x += GetFrameTime() * 2;
    if (IsKeyDown(KEY_D)) c_x -= GetFrameTime() * 2;
    if (IsKeyDown(KEY_W)) c_y += GetFrameTime() * 2;
    if (IsKeyDown(KEY_S)) c_y -= GetFrameTime() * 2;
  
    camera.position = (Vector3){
      cameraDistance * cos(c_x),
      c_y * cameraDistance,
      cameraDistance * sin(c_x)
    };
    camera.position = Vector3Add(camera.position, target);

    BeginDrawing();
    ClearBackground(BLACK);

    UpdateCamera(&camera, CAMERA_PERSPECTIVE);
    BeginMode3D(camera);
    
    const float scale = 1.5;
    for (size_t i = 0; i < CUBE_SIZE*scale; i++) {
      for (size_t j = 0; j < CUBE_SIZE*scale; j++) {
        for (size_t k = 0; k < CUBE_SIZE*scale; k++) {
          MAT_AT(NN_INPUT(nn), 0, 0) = i / (CUBE_SIZE*scale - 1.0);
          MAT_AT(NN_INPUT(nn), 0, 1) = j / (CUBE_SIZE*scale - 1.0);
          MAT_AT(NN_INPUT(nn), 0, 2) = k / (CUBE_SIZE*scale - 1.0);
          MAT_AT(NN_INPUT(nn), 0, 3) = slider;
          nn_forward(nn);

          if (MAT_AT(NN_OUTPUT(nn), 0, 0) > 0.5) {
            DrawCubeV((Vector3){j/scale, k/scale, i/scale}, (Vector3){1/scale, 1/scale, 1/scale}, BLACK);
            float t = 0.05/scale;
            DrawCubeWiresV((Vector3){j/scale + t / 2.0, k/scale + t / 2.0, i/scale + t / 2.0},
                           (Vector3){1/scale + t, 1/scale + t, 1/scale + t}, WHITE);
          }
        }
      }
    }

    draw_voxels(torus, (Vector3){CUBE_SIZE + 5.0, 0, 0});
    draw_voxels(apple, (Vector3){-(CUBE_SIZE + 5.0), 0, 0});

    EndMode3D();

    DrawRectangle(slider*screenWidth-15, screenHeight-30, 30, 30, RED);

    EndDrawing();
  }

  free(torus);
  free(apple);
  nn_free(nn);
  nn_free(g);

  CloseWindow();
  return 0;
}
