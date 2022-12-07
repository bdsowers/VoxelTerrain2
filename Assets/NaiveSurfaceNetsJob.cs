using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Profiling;
using Unity.Jobs;
using Unity.Burst;
using Unity.Collections;

[BurstCompile(CompileSynchronously = true)]
public struct NaiveSurfaceNetsJob : IJob
{
    [ReadOnly] public NativeArray<float> sdf;
    [WriteOnly] public NativeArray<Vector3> positions;
    [WriteOnly] public NativeArray<Vector3> normals;
    [WriteOnly] public NativeArray<Vector3Int> surfacePoints;
    [WriteOnly] public NativeArray<int> surfaceStrides;
    [WriteOnly] public NativeArray<int> strideToIndex;
    [WriteOnly] public NativeArray<int> counts;

    public NativeArray<bool> dirty;
    public NativeArray<bool> cachedOnSurface;

    [ReadOnly] public NativeArray<Vector3Int> CUBE_CORNERS;
    [ReadOnly] public NativeArray<Vector2Int> CUBE_EDGES;
    [ReadOnly] public NativeArray<Vector3> CUBE_CORNER_VECTORS;
    [ReadOnly] public Vector3Int CELL_SIZE;

    public Vector3Int min, max;
    public NativeArray<float> cornerDists;

    public void Execute()
    {
        int numPositions = 0;
        int numSurfacePoints = 0;

        for (int z = min.z; z <= max.z; ++z)
        {
            for (int y = min.y; y <= max.y; ++y)
            {
                for (int x = min.x; x <= max.x; ++x)
                {
                    int stride = x + y * CELL_SIZE.y + z * CELL_SIZE.x * CELL_SIZE.y;
                    Vector3 p = new Vector3(x, y, z);
                    int prevNumPositions = numPositions;

                    if (dirty[stride] || cachedOnSurface[stride])
                    {
                        numPositions = EstimateSurfaceInCube(p, stride, numPositions);
                        cachedOnSurface[stride] = numPositions != prevNumPositions;
                        dirty[stride] = false;
                    }

                    if (cachedOnSurface[stride])
                    {
                        strideToIndex[stride] = numPositions - 1;
                        surfacePoints[numSurfacePoints] = new Vector3Int(x, y, z);
                        surfaceStrides[numSurfacePoints] = stride;
                        ++numSurfacePoints;
                    }
                    else
                    {
                        strideToIndex[stride] = -1;
                    }
                }
            }
        }

        counts[0] = numPositions;
        counts[1] = numSurfacePoints;
    }

    private int EstimateSurfaceInCube(Vector3 p, int minCornerStride, int numPositions)
    {
        int numNegatives = 0;

        for (int i = 0; i < 8; ++i)
        {
            int x = CUBE_CORNERS[i].x, y = CUBE_CORNERS[i].y, z = CUBE_CORNERS[i].z;
            int cornerLinearized = x + y * CELL_SIZE.y + z * CELL_SIZE.x * CELL_SIZE.y;

            int stride = minCornerStride + cornerLinearized;
            if (stride < 0 || stride >= sdf.Length)
            {
                continue;
            }

            float d = sdf[stride];
            cornerDists[i] = d;
            if (d < 0f)
            {
                numNegatives++;
            }
        }

        if (numNegatives == 0 || numNegatives == 8)
        {
            return numPositions;
        }

        Vector3 c = CentroidOfEdgeIntersection(ref cornerDists);
        positions[numPositions] = (p + c);
        normals[numPositions] = SDFGradient(ref cornerDists, c);
        ++numPositions;

        return numPositions;
    }

    Vector3 CentroidOfEdgeIntersection(ref NativeArray<float> cornerDists)
    {
        int count = 0;
        Vector3 sum = Vector3.zero;

        foreach (Vector2Int corner in CUBE_EDGES)
        {
            float d1 = cornerDists[corner.x];
            float d2 = cornerDists[corner.y];
            if (!Mathf.Approximately(Mathf.Sign(d1), Mathf.Sign(d2)))
            {
                count++;
                sum += EstimateSurfaceEdgeIntersection(corner.x, corner.y, d1, d2);
            }
        }

        return sum / count;
    }

    Vector3 EstimateSurfaceEdgeIntersection(int corner1, int corner2, float value1, float value2)
    {
        float interp = value1 / (value1 - value2);
        return (1 - interp) * CUBE_CORNER_VECTORS[corner1] + interp * CUBE_CORNER_VECTORS[corner2];
    }

    public Vector3 SDFGradient(ref NativeArray<float> dists, Vector3 s)
    {
        Vector3 p00 = new Vector3(dists[0b001], dists[0b010], dists[0b100]);
        Vector3 n00 = new Vector3(dists[0b000], dists[0b000], dists[0b000]);

        Vector3 p10 = new Vector3(dists[0b101], dists[0b011], dists[0b110]);
        Vector3 n10 = new Vector3(dists[0b100], dists[0b001], dists[0b010]);

        Vector3 p01 = new Vector3(dists[0b011], dists[0b110], dists[0b101]);
        Vector3 n01 = new Vector3(dists[0b010], dists[0b100], dists[0b001]);

        Vector3 p11 = new Vector3(dists[0b111], dists[0b111], dists[0b111]);
        Vector3 n11 = new Vector3(dists[0b110], dists[0b101], dists[0b011]);

        // Each dimension encodes an edge delta, giving 12 in total.
        Vector3 d00 = p00 - n00; // Edges (0b00x, 0b0y0, 0bz00)
        Vector3 d10 = p10 - n10; // Edges (0b10x, 0b0y1, 0bz10)
        Vector3 d01 = p01 - n01; // Edges (0b01x, 0b1y0, 0bz01)
        Vector3 d11 = p11 - n11; // Edges (0b11x, 0b1y1, 0bz11)

        Vector3 neg = Vector3.one - s;

        // Do bilinear interpolation between 4 edges in each dimension.
        Vector3 result = ElementMul(YZX(neg), ZXY(neg), d00)
            + ElementMul(YZX(neg), ZXY(s), d10)
            + ElementMul(YZX(s), ZXY(neg), d01)
            + ElementMul(YZX(s), ZXY(s), d11);
        return result.normalized;
    }

    private Vector3 ElementMul(Vector3 v1, Vector3 v2, Vector3 v3)
    {
        return new Vector3(v1.x * v2.x * v3.x, v1.y * v2.y * v3.y, v1.z * v2.z * v3.z);
    }

    private Vector3 YZX(Vector3 vec)
    {
        return new Vector3(vec.y, vec.z, vec.x);
    }

    private Vector3 ZXY(Vector3 vec)
    {
        return new Vector3(vec.z, vec.x, vec.y);
    }
}