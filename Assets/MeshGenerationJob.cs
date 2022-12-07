using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Profiling;
using Unity.Jobs;
using Unity.Burst;
using Unity.Collections;

public struct MeshGenerationJob : IJob
{
    [ReadOnly] public NativeArray<float> sdf;
    [ReadOnly] public NativeArray<int> LINEARIZED_XYZ_STRIDES;
    [ReadOnly] public NativeArray<int> counts;
    [ReadOnly] public NativeArray<Vector3Int> surfacePoints;
    [ReadOnly] public NativeArray<int> surfaceStrides;
    [ReadOnly] public NativeArray<int> strideToIndex;
    [ReadOnly] public Vector3Int min;
    [ReadOnly] public Vector3Int max;
    [ReadOnly] public NativeArray<Vector3> positions;

    [WriteOnly] public NativeArray<int> indices;
    [WriteOnly] public NativeArray<int> numIndices;

    private int indicesCounter;

    public void Execute()
    {
        MakeAllQuads();
        numIndices[0] = indicesCounter;
    }

    private void MakeAllQuads()
    {
        int numSurfacePoints = counts[1];
        for (int i = 0; i < numSurfacePoints; ++i)
        {
            Vector3Int surfacePoint = surfacePoints[i];
            int p_stride = surfaceStrides[i];

            if (surfacePoint.y != min.y && surfacePoint.z != min.z && surfacePoint.x != max.x - 1)
            {
                MaybeMakeQuad(
                    p_stride,
                    p_stride + LINEARIZED_XYZ_STRIDES[0],
                    LINEARIZED_XYZ_STRIDES[1],
                    LINEARIZED_XYZ_STRIDES[2]);
            }

            if (surfacePoint.x != min.x && surfacePoint.z != min.z && surfacePoint.y != max.y - 1)
            {
                MaybeMakeQuad(
                    p_stride,
                    p_stride + LINEARIZED_XYZ_STRIDES[1],
                    LINEARIZED_XYZ_STRIDES[2],
                    LINEARIZED_XYZ_STRIDES[0]);
            }

            if (surfacePoint.x != min.x && surfacePoint.y != min.y && surfacePoint.z != max.z - 1)
            {
                MaybeMakeQuad(
                    p_stride,
                    p_stride + LINEARIZED_XYZ_STRIDES[2],
                    LINEARIZED_XYZ_STRIDES[0],
                    LINEARIZED_XYZ_STRIDES[1]);
            }
        }
    }

    private void MaybeMakeQuad(int p1, int p2, int axis_b_stride, int axis_c_stride)
    {
        float d1 = sdf[p1];
        float d2 = sdf[p2];

        bool negativeFace = false;
        if (d1 < 0f && d2 >= 0f) negativeFace = false;
        else if (d1 >= 0f && d2 < 0f) negativeFace = true;
        else return;

        int v1 = strideToIndex[p1];
        int v2 = strideToIndex[p1 - axis_b_stride];
        int v3 = strideToIndex[p1 - axis_c_stride];
        int v4 = strideToIndex[p1 - axis_b_stride - axis_c_stride];

        if (Vector3.SqrMagnitude(positions[v1] - positions[v4]) < Vector3.SqrMagnitude(positions[v2] - positions[v3]))
        {
            if (negativeFace)
            {
                // [v1, v4, v2, v1, v3, v4]
                indices[indicesCounter++] = v1;
                indices[indicesCounter++] = v4;
                indices[indicesCounter++] = v2;

                indices[indicesCounter++] = v1;
                indices[indicesCounter++] = v3;
                indices[indicesCounter++] = v4;
            }
            else
            {
                // [v1, v2, v4, v1, v4, v3]
                indices[indicesCounter++] = v1;
                indices[indicesCounter++] = v2;
                indices[indicesCounter++] = v4;

                indices[indicesCounter++] = v1;
                indices[indicesCounter++] = v4;
                indices[indicesCounter++] = v3;
            }
        }
        else if (negativeFace)
        {
            // [v2, v3, v4, v2, v1, v3]
            indices[indicesCounter++] = v2;
            indices[indicesCounter++] = v3;
            indices[indicesCounter++] = v4;

            indices[indicesCounter++] = v2;
            indices[indicesCounter++] = v1;
            indices[indicesCounter++] = v3;
        }
        else
        {
            // [v2, v4, v3, v2, v3, v1]
            indices[indicesCounter++] = v2;
            indices[indicesCounter++] = v4;
            indices[indicesCounter++] = v3;

            indices[indicesCounter++] = v2;
            indices[indicesCounter++] = v3;
            indices[indicesCounter++] = v1;
        }
    }

    private void GenerateMesh(int numPositions, ref NativeArray<Vector3> positions, int numIndices, ref NativeArray<int> indices, ref NativeArray<Vector3> normals)
    {
        /*
        Vector3[] verticesCpy = new Vector3[numPositions];
        Vector3[] normalsCpy = new Vector3[numPositions];
        int[] indicesCpy = new int[numIndices];
        NativeArray<Vector3>.Copy(positions, verticesCpy, numPositions);
        NativeArray<Vector3>.Copy(normals, normalsCpy, numPositions);
        NativeArray<int>.Copy(indices, indicesCpy, numIndices);

        Mesh mesh = new Mesh();
        mesh.vertices = verticesCpy;
        NativeArray<Vector3>.Copy(positions, mesh.vertices, numPositions);
        mesh.SetIndices(indicesCpy, MeshTopology.Triangles, 0);
        mesh.normals = normalsCpy;
        mesh.UploadMeshData(false);
        mesh.RecalculateBounds();

        GetComponent<MeshFilter>().mesh = mesh;
        GetComponent<MeshCollider>().sharedMesh = mesh;
        */
    }
}
