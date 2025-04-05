#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    int colorIndex;
} Voxel;

typedef struct {
    uint8_t x, y, z, colorIndex;
} FileVoxel;

typedef struct {
    char id[5];
    uint32_t contentSize;
    uint32_t childrenSize;
} ChunkHeader;

typedef struct {
    int32_t sizeX, sizeY, sizeZ;
} VoxSize;

static void read_chunk_header(FILE *fp, ChunkHeader *header) {
    fread(header->id, 1, 4, fp);
    header->id[4] = '\0';
    fread(&header->contentSize, sizeof(uint32_t), 1, fp);
    fread(&header->childrenSize, sizeof(uint32_t), 1, fp);
}

static size_t index3D(int x, int y, int z) {
    return (size_t)x + (size_t)y * CUBE_SIZE + (size_t)z * CUBE_SIZE * CUBE_SIZE;
}

Voxel* load_vox_model(const char *filepath) {
    FILE *fp = fopen(filepath, "rb");
    if (!fp) {
        perror("Error opening file");
        return NULL;
    }

    char magic[5];
    if (fread(magic, 1, 4, fp) != 4) {
        fprintf(stderr, "Error reading file header.\n");
        fclose(fp);
        return NULL;
    }
    magic[4] = '\0';
    if (strcmp(magic, "VOX ") != 0) {
        fprintf(stderr, "Not a valid VOX file.\n");
        fclose(fp);
        return NULL;
    }

    uint32_t version;
    fread(&version, sizeof(uint32_t), 1, fp);

    ChunkHeader mainChunk;
    read_chunk_header(fp, &mainChunk);
    if (strcmp(mainChunk.id, "MAIN") != 0) {
        fprintf(stderr, "Expected MAIN chunk, found %s\n", mainChunk.id);
        fclose(fp);
        return NULL;
    }

    Voxel *grid = malloc(CUBE_SIZE * CUBE_SIZE * CUBE_SIZE * sizeof(Voxel));
    if (!grid) {
        fprintf(stderr, "Memory allocation error for grid\n");
        fclose(fp);
        return NULL;
    }

    for (int i = 0; i < CUBE_SIZE * CUBE_SIZE * CUBE_SIZE; i++) {
        grid[i].colorIndex = -1;
    }

    VoxSize modelSize = {0, 0, 0};
    int modelLoaded = 0;

    long mainChunkEnd = ftell(fp) + mainChunk.childrenSize;
    while (ftell(fp) < mainChunkEnd) {
        ChunkHeader chunk;
        read_chunk_header(fp, &chunk);

        long chunkContentPos = ftell(fp);

        if (strcmp(chunk.id, "SIZE") == 0) {
            fread(&modelSize.sizeX, sizeof(int32_t), 1, fp);
            fread(&modelSize.sizeY, sizeof(int32_t), 1, fp);
            fread(&modelSize.sizeZ, sizeof(int32_t), 1, fp);
            modelLoaded = 1;
        }
        else if (strcmp(chunk.id, "XYZI") == 0) {
            if (!modelLoaded) {
                fprintf(stderr, "XYZI chunk encountered before SIZE chunk.\n");
                free(grid);
                fclose(fp);
                return NULL;
            }

            int offsetX = (CUBE_SIZE - modelSize.sizeX) / 2;
            int offsetY = (CUBE_SIZE - modelSize.sizeY) / 2;
            int offsetZ = (CUBE_SIZE - modelSize.sizeZ) / 2;

            // Read the number of voxels.
            uint32_t numVoxels;
            fread(&numVoxels, sizeof(uint32_t), 1, fp);

            // Allocate temporary buffer for voxels from the file.
            FileVoxel *fileVoxels = malloc(numVoxels * sizeof(FileVoxel));
            if (!fileVoxels) {
                fprintf(stderr, "Memory allocation error for voxels\n");
                free(grid);
                fclose(fp);
                return NULL;
            }
            fread(fileVoxels, sizeof(FileVoxel), numVoxels, fp);

            for (uint32_t i = 0; i < numVoxels; i++) {
                int gridX = fileVoxels[i].x + offsetX;
                int gridY = fileVoxels[i].y + offsetY;
                int gridZ = fileVoxels[i].z + offsetZ;
                if (gridX < 0 || gridX >= CUBE_SIZE ||
                    gridY < 0 || gridY >= CUBE_SIZE ||
                    gridZ < 0 || gridZ >= CUBE_SIZE) {
                    fprintf(stderr, "Voxel %u at (%d, %d, %d) is out of bounds after centering.\n",
                            i, gridX, gridY, gridZ);
                    continue;
                }
                size_t idx = index3D(gridX, gridY, gridZ);
                grid[idx].colorIndex = fileVoxels[i].colorIndex;
            }
            free(fileVoxels);
        }
        else {
            fseek(fp, chunk.contentSize, SEEK_CUR);
        }
        if (chunk.childrenSize > 0) {
            fseek(fp, chunk.childrenSize, SEEK_CUR);
        }

        long expectedPos = chunkContentPos + chunk.contentSize + chunk.childrenSize;
        fseek(fp, expectedPos, SEEK_SET);
    }

    fclose(fp);
    return grid;
}
