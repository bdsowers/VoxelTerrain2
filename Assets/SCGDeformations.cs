using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Profiling;
using Unity.Collections;

public static class SCGDeformations
{
    public static void RenderSphereIntoChunk(Vector3Int center, Vector3Int CELL_SIZE, float radius, ref NativeArray<float> sdf, ref NativeArray<bool> dirty, bool subtract)
    {
        Profiler.BeginSample("Rendering sphere into chunk");

        Vector3 min = center - Vector3.one * (radius + 1);
        Vector3 max = center + Vector3.one * (radius + 1);
        Vector3Int minInt = new Vector3Int(Mathf.FloorToInt(min.x), Mathf.FloorToInt(min.y), Mathf.FloorToInt(min.z));
        Vector3Int maxInt = new Vector3Int(Mathf.CeilToInt(max.x), Mathf.CeilToInt(max.y), Mathf.CeilToInt(max.z));

        minInt.x = Mathf.Max(minInt.x, 1);
        minInt.y = Mathf.Max(minInt.y, 1);
        minInt.z = Mathf.Max(minInt.z, 1);

        maxInt.x = Mathf.Min(maxInt.x, CELL_SIZE.x - 1);
        maxInt.y = Mathf.Min(maxInt.y, CELL_SIZE.y - 1);
        maxInt.z = Mathf.Min(maxInt.z, CELL_SIZE.z - 1);

        for (int x = minInt.x; x < maxInt.x; x++)
        {
            for (int y = minInt.y; y < maxInt.y; y++)
            {
                for (int z = minInt.z; z < maxInt.z; z++)
                {
                    Vector3 pos = new Vector3(x, y, z);
                    float val = Vector3.Distance(pos, center) - radius;

                    int lin = x + y * CELL_SIZE.y + z * CELL_SIZE.x * CELL_SIZE.y;
                    float curr = sdf[lin];

                    if (subtract)
                        sdf[lin] = Mathf.Max(-val, curr);
                    else
                        sdf[lin] = Mathf.Min(curr, val);

                    dirty[lin] = true;
                }
            }
        }

        Profiler.EndSample();
    }

    public static void RenderCubeIntoChunk(Vector3Int center, Vector3Int CELL_SIZE, Vector3 size, ref NativeArray<float> sdf, ref NativeArray<bool> dirty)
    {
        Vector3 min = center - size / 1.5f;
        Vector3 max = center + size / 1.5f;
        Vector3Int minInt = new Vector3Int(Mathf.FloorToInt(min.x), Mathf.FloorToInt(min.y), Mathf.FloorToInt(min.z));
        Vector3Int maxInt = new Vector3Int(Mathf.CeilToInt(max.x), Mathf.CeilToInt(max.y), Mathf.CeilToInt(max.z));

        minInt.x = Mathf.Max(minInt.x, 1);
        minInt.y = Mathf.Max(minInt.y, 1);
        minInt.z = Mathf.Max(minInt.z, 1);

        maxInt.x = Mathf.Min(maxInt.x, CELL_SIZE.x - 1);
        maxInt.y = Mathf.Min(maxInt.y, CELL_SIZE.y - 1);
        maxInt.z = Mathf.Min(maxInt.z, CELL_SIZE.z - 1);

        for (int x = minInt.x; x < maxInt.x; x++)
        {
            for (int y = minInt.y; y < maxInt.y; y++)
            {
                for (int z = minInt.z; z < maxInt.z; z++)
                {
                    Vector3 pos = new Vector3(x, y, z);
                    float val = (pos.x > center.x - size.x / 2 && pos.x < center.x + size.x / 2 &&
                                pos.y > center.y - size.y / 2 && pos.y < center.y + size.y / 2 &&
                                pos.z > center.z - size.z / 2 && pos.z < center.z + size.z / 2) ? -10f : 10f;

                    int lin = x + y * CELL_SIZE.x + z * CELL_SIZE.x * CELL_SIZE.y;
                    float curr = sdf[lin];
                    sdf[lin] = Mathf.Min(curr, val);

                    dirty[lin] = true;
                }
            }
        }
    }
}
