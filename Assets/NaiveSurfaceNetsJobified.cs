using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Profiling;
using Unity.Jobs;
using Unity.Burst;
using Unity.Collections;

public class NaiveSurfaceNetsJobified : MonoBehaviour
{
    public readonly static int CHUNK_SIZE = 64;
    public readonly static int CHUNK_DATA_LENGTH = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
    public readonly static Vector3Int CELL_SIZE = new Vector3Int(CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE);

    private int[] LINEARIZED_XYZ_STRIDES = null;

    private NativeArray<float> sdf;
    private NativeArray<Vector3> positions;
    private NativeArray<Vector3> normals;
    private NativeArray<int> indices;
    private NativeArray<Vector3Int> surfacePoints;
    private NativeArray<int> surfaceStrides;
    private NativeArray<int> strideToIndex;
    private NativeArray<int> counts;
    private NativeArray<float> cornerDists;
    private NativeArray<Vector3Int> CUBE_CORNERS_NATIVE;
    private NativeArray<Vector2Int> CUBE_EDGES_NATIVE;
    private NativeArray<Vector3> CUBE_CORNER_VECTORS_NATIVE;
    
    private NativeArray<bool> dirty;
    private NativeArray<bool> cachedOnSurface;

    private readonly static List<Vector3Int> CUBE_CORNERS = new List<Vector3Int>()
    {
        new Vector3Int(0, 0, 0),
        new Vector3Int(1, 0, 0),
        new Vector3Int(0, 1, 0),
        new Vector3Int(1, 1, 0),
        new Vector3Int(0, 0, 1),
        new Vector3Int(1, 0, 1),
        new Vector3Int(0, 1, 1),
        new Vector3Int(1, 1, 1),
    };

    private readonly static List<Vector2Int> CUBE_EDGES = new List<Vector2Int>()
    {
        new Vector2Int(0b000, 0b001),
        new Vector2Int(0b000, 0b010),
        new Vector2Int(0b000, 0b100),
        new Vector2Int(0b001, 0b011),
        new Vector2Int(0b001, 0b101),
        new Vector2Int(0b010, 0b011),
        new Vector2Int(0b010, 0b110),
        new Vector2Int(0b011, 0b111),
        new Vector2Int(0b100, 0b101),
        new Vector2Int(0b100, 0b110),
        new Vector2Int(0b101, 0b111),
        new Vector2Int(0b110, 0b111),
    };

    private readonly static List<Vector3> CUBE_CORNER_VECTORS = new List<Vector3>()
    {
        new Vector3(0.0f, 0.0f, 0.0f),
        new Vector3(1.0f, 0.0f, 0.0f),
        new Vector3(0.0f, 1.0f, 0.0f),
        new Vector3(1.0f, 1.0f, 0.0f),
        new Vector3(0.0f, 0.0f, 1.0f),
        new Vector3(1.0f, 0.0f, 1.0f),
        new Vector3(0.0f, 1.0f, 1.0f),
        new Vector3(1.0f, 1.0f, 1.0f),
    };

    NaiveSurfaceNetsJob job;
    JobHandle jobHandle;

    public void Start()
    {
        LINEARIZED_XYZ_STRIDES = new int[]
        {
            Linearize(1, 0, 0),
            Linearize(0, 1, 0),
            Linearize(0, 0, 1),
        };

        sdf = new NativeArray<float>(CHUNK_DATA_LENGTH, Allocator.Persistent);
        positions = new NativeArray<Vector3>(CHUNK_DATA_LENGTH, Allocator.Persistent);
        normals = new NativeArray<Vector3>(CHUNK_DATA_LENGTH, Allocator.Persistent);
        dirty = new NativeArray<bool>(CHUNK_DATA_LENGTH, Allocator.Persistent);
        cachedOnSurface = new NativeArray<bool>(CHUNK_DATA_LENGTH, Allocator.Persistent);
        indices = new NativeArray<int>(CHUNK_DATA_LENGTH, Allocator.Persistent);
        surfacePoints = new NativeArray<Vector3Int>(CHUNK_DATA_LENGTH, Allocator.Persistent);
        surfaceStrides = new NativeArray<int>(CHUNK_DATA_LENGTH, Allocator.Persistent);
        strideToIndex = new NativeArray<int>(CHUNK_DATA_LENGTH, Allocator.Persistent);
        counts = new NativeArray<int>(2, Allocator.Persistent);
        cornerDists = new NativeArray<float>(8, Allocator.Persistent);
        CUBE_CORNERS_NATIVE = new NativeArray<Vector3Int>(CUBE_CORNERS.Count, Allocator.Persistent);
        for (int i = 0; i < CUBE_CORNERS.Count; ++i) CUBE_CORNERS_NATIVE[i] = CUBE_CORNERS[i];
        CUBE_CORNER_VECTORS_NATIVE = new NativeArray<Vector3>(CUBE_CORNER_VECTORS.Count, Allocator.Persistent);
        for (int i = 0; i < CUBE_CORNER_VECTORS.Count; ++i) CUBE_CORNER_VECTORS_NATIVE[i] = CUBE_CORNER_VECTORS[i];
        CUBE_EDGES_NATIVE = new NativeArray<Vector2Int>(CUBE_EDGES.Count, Allocator.Persistent);
        for (int i = 0; i < CUBE_EDGES.Count; ++i) CUBE_EDGES_NATIVE[i] = CUBE_EDGES[i];

        for (int i = 0; i < CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE; ++i)
        {
            sdf[i] = 10;
            dirty[i] = true;
        }

        RenderSphereIntoChunk(new Vector3Int(25, 25, 25), 10f, ref sdf, false);
        RenderSphereIntoChunk(new Vector3Int(35, 25, 25), 10f, ref sdf, false);

        BeginJob();
        CompleteJob();

        Debug.Log("We should see something now...");
    }

    private void CompleteJob()
    {
        jobHandle.Complete();

        int numIndices = 0;
        MakeAllQuads(job.counts[1], ref sdf, ref job.surfacePoints, ref job.surfaceStrides, ref job.strideToIndex, ref job.positions, ref indices, ref numIndices, job.min, job.max);
        GenerateMesh(job.counts[0], ref job.positions, numIndices, ref indices, ref job.normals);
    }

    private void BeginJob()
    {
        job = new NaiveSurfaceNetsJob();
        job.sdf = sdf;
        job.positions = positions;
        job.normals = normals;
        job.surfacePoints = surfacePoints;
        job.surfaceStrides = surfaceStrides;
        job.strideToIndex = strideToIndex;
        job.min = new Vector3Int(1, 1, 1);
        job.max = new Vector3Int(CHUNK_SIZE - 2, CHUNK_SIZE - 2, CHUNK_SIZE - 2);
        job.counts = counts;
        job.cornerDists = cornerDists;
        job.dirty = dirty;
        job.cachedOnSurface = cachedOnSurface;
        job.CELL_SIZE = CELL_SIZE;
        job.CUBE_CORNERS = CUBE_CORNERS_NATIVE;
        job.CUBE_CORNER_VECTORS = CUBE_CORNER_VECTORS_NATIVE;
        job.CUBE_EDGES = CUBE_EDGES_NATIVE;

        jobHandle = job.Schedule();
    }

    private void Update()
    {
        if (Input.GetMouseButtonDown(0) || Input.GetMouseButtonDown(1))
        {
            Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
            if (Physics.Raycast(ray, out RaycastHit hitInfo, float.MaxValue))
            {
                StartCoroutine(AnimateBlobGrow(hitInfo.point, Input.GetMouseButtonDown(1)));
            }
        }
    }

    private int Linearize(int x, int y, int z)
    {
        return x + y * CELL_SIZE.y + z * CELL_SIZE.x * CELL_SIZE.y;
    }

    private IEnumerator AnimateBlobGrow(Vector3 pos, bool subtract)
    {
        Vector3Int posInt = new Vector3Int(Mathf.CeilToInt(pos.x), Mathf.CeilToInt(pos.y), Mathf.CeilToInt(pos.z));
        for (float i = 1.0f; i <= 5f; i += Time.deltaTime * 15f)
        {
            if (!jobHandle.IsCompleted) yield return null;

            CompleteJob();

            RenderSphereIntoChunk(posInt, i, ref sdf, subtract);

            BeginJob();

            yield return null;
        }
    }

    void RenderSphereIntoChunk(Vector3Int center, float radius, ref NativeArray<float> sdf, bool subtract)
    {
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
    }

    private void MakeAllQuads(int numSurfacePoints, ref NativeArray<float> sdf, ref NativeArray<Vector3Int> surfacePoints, ref NativeArray<int> surfaceStrides, ref NativeArray<int> strideToIndex, ref NativeArray<Vector3> positions, ref NativeArray<int> indices, ref int numIndices, Vector3Int min, Vector3Int max)
    {
        for (int i = 0; i < numSurfacePoints; ++i)
        {
            Vector3Int surfacePoint = surfacePoints[i];
            int p_stride = surfaceStrides[i];

            if (surfacePoint.y != min.y && surfacePoint.z != min.z && surfacePoint.x != max.x - 1)
            {
                MaybeMakeQuad(ref sdf,
                    ref strideToIndex,
                    ref positions,
                    p_stride,
                    p_stride + LINEARIZED_XYZ_STRIDES[0],
                    LINEARIZED_XYZ_STRIDES[1],
                    LINEARIZED_XYZ_STRIDES[2],
                    ref indices,
                    ref numIndices);
            }

            if (surfacePoint.x != min.x && surfacePoint.z != min.z && surfacePoint.y != max.y - 1)
            {
                MaybeMakeQuad(ref sdf,
                    ref strideToIndex,
                    ref positions,
                    p_stride,
                    p_stride + LINEARIZED_XYZ_STRIDES[1],
                    LINEARIZED_XYZ_STRIDES[2],
                    LINEARIZED_XYZ_STRIDES[0],
                    ref indices,
                    ref numIndices);
            }

            if (surfacePoint.x != min.x && surfacePoint.y != min.y && surfacePoint.z != max.z - 1)
            {
                MaybeMakeQuad(ref sdf,
                    ref strideToIndex,
                    ref positions,
                    p_stride,
                    p_stride + LINEARIZED_XYZ_STRIDES[2],
                    LINEARIZED_XYZ_STRIDES[0],
                    LINEARIZED_XYZ_STRIDES[1],
                    ref indices,
                    ref numIndices);
            }
        }
    }

    private void MaybeMakeQuad(ref NativeArray<float> sdf, ref NativeArray<int> strideToIndex, ref NativeArray<Vector3> positions, int p1, int p2, int axis_b_stride, int axis_c_stride, ref NativeArray<int> indices, ref int numIndices)
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
                indices[numIndices++] = v1;
                indices[numIndices++] = v4;
                indices[numIndices++] = v2;

                indices[numIndices++] = v1;
                indices[numIndices++] = v3;
                indices[numIndices++] = v4;
            }
            else
            {
                // [v1, v2, v4, v1, v4, v3]
                indices[numIndices++] = v1;
                indices[numIndices++] = v2;
                indices[numIndices++] = v4;

                indices[numIndices++] = v1;
                indices[numIndices++] = v4;
                indices[numIndices++] = v3;
            }
        }
        else if (negativeFace)
        {
            // [v2, v3, v4, v2, v1, v3]
            indices[numIndices++] = v2;
            indices[numIndices++] = v3;
            indices[numIndices++] = v4;

            indices[numIndices++] = v2;
            indices[numIndices++] = v1;
            indices[numIndices++] = v3;
        }
        else
        {
            // [v2, v4, v3, v2, v3, v1]
            indices[numIndices++] = v2;
            indices[numIndices++] = v4;
            indices[numIndices++] = v3;

            indices[numIndices++] = v2;
            indices[numIndices++] = v3;
            indices[numIndices++] = v1;
        }
    }

    private void GenerateMesh(int numPositions, ref NativeArray<Vector3> positions, int numIndices, ref NativeArray<int> indices, ref NativeArray<Vector3> normals)
    {
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
    }

    [BurstCompile(CompileSynchronously = true)]
    private struct NaiveSurfaceNetsJob : IJob
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
}
