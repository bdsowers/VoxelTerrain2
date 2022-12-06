using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Profiling;
using Unity.Mathematics;

public class NaiveSurfaceNets : MonoBehaviour
{
    
    private const int CHUNK_SIZE = 128;

    private Vector3Int CELL_SIZE = new Vector3Int(CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE);
    private const int CHUNK_DATA_LENGTH = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

    float[] sdf = new float[CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE];

    private bool anythingDirty = false;
    bool[] dirty = new bool[CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE];
    bool[] cachedOnSurface = new bool[CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE];
    
    float[] cornerDists = new float[8];
    WorkingData workingData = new WorkingData();

    private int Linearize(Vector3Int point)
    {
        int result =  point.x + point.y * CELL_SIZE.x + point.z * CELL_SIZE.x * CELL_SIZE.y;
        return result;
    }

    private int Linearize(int x, int y, int z)
    {
        return x + y * CELL_SIZE.y + z * CELL_SIZE.x * CELL_SIZE.y;
    }

    private void EstimateSurface(float[] sdf, Vector3Int min, Vector3Int max, WorkingData workingData)
    {
        for (int z = min.z; z <= max.z; ++z)
        {
            for (int y = min.y; y <= max.y; ++y)
            {
                for (int x = min.x; x <= max.x; ++x)
                {
                    int stride = x + y * CELL_SIZE.x + z * CELL_SIZE.x * CELL_SIZE.y;
                    Vector3 p = new Vector3(x, y, z);

                    if (dirty[stride])
                    {
                        cachedOnSurface[stride] = EstimateSurfaceInCube(sdf, p, stride, workingData);
                        dirty[stride] = false;
                    }
                    else if (cachedOnSurface[stride])
                    {
                        // TODO : We could potentially optimize this away with more caching?
                        cachedOnSurface[stride] = EstimateSurfaceInCube(sdf, p, stride, workingData);
                    }

                    if (cachedOnSurface[stride])
                    {
                        workingData.strideToIndex[stride] = workingData.positions.Count - 1;
                        workingData.surfacePoints.Add(new Vector3Int(x, y, z));
                        workingData.surfaceStrides.Add(stride);
                    }
                }
            }
        }
    }

    private bool EstimateSurfaceInCube(float[] sdf, Vector3 p, int minCornerStride, WorkingData workingData)
    {
        int numNegatives = 0;

        for (int i = 0; i < 8; ++i)
        {
            int stride = minCornerStride + LINEARIZED_CUBE_CORNERS[i];
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
            return false;
        }

        Vector3 c = CentroidOfEdgeIntersection(cornerDists);
        workingData.positions.Add(p + c);
        workingData.normals.Add(SDFGradient(cornerDists, c));
        return true;
    }

    Vector3 CentroidOfEdgeIntersection(float[] cornerDists)
    {
        int count = 0;
        Vector3 sum = Vector3.zero;

        foreach(Vector2Int corner in CUBE_EDGES)
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

    public Vector3 SDFGradient(float[] dists, Vector3 s)
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

    private void MakeAllQuads(float[] sdf, Vector3Int min, Vector3Int max, WorkingData workingData)
    {
        for (int i = 0; i < workingData.surfacePoints.Count; ++i)
        {
            Vector3Int surfacePoint = workingData.surfacePoints[i];
            int p_stride = workingData.surfaceStrides[i];

            if (surfacePoint.y != min.y && surfacePoint.z != min.z && surfacePoint.x != max.x - 1)
            {
                MaybeMakeQuad(sdf, 
                    workingData.strideToIndex, 
                    workingData.positions, 
                    p_stride, 
                    p_stride + LINEARIZED_XYZ_STRIDES[0],
                    LINEARIZED_XYZ_STRIDES[1],
                    LINEARIZED_XYZ_STRIDES[2], 
                    workingData.indices);
            }

            if (surfacePoint.x != min.x && surfacePoint.z != min.z && surfacePoint.y != max.y - 1)
            {
                MaybeMakeQuad(sdf,
                    workingData.strideToIndex,
                    workingData.positions,
                    p_stride,
                    p_stride + LINEARIZED_XYZ_STRIDES[1],
                    LINEARIZED_XYZ_STRIDES[2],
                    LINEARIZED_XYZ_STRIDES[0],
                    workingData.indices);
            }

            if (surfacePoint.x != min.x && surfacePoint.y != min.y && surfacePoint.z != max.z - 1)
            {
                MaybeMakeQuad(sdf,
                    workingData.strideToIndex,
                    workingData.positions,
                    p_stride,
                    p_stride + LINEARIZED_XYZ_STRIDES[2],
                    LINEARIZED_XYZ_STRIDES[0],
                    LINEARIZED_XYZ_STRIDES[1],
                    workingData.indices);
            }
        }
    }

    private void MaybeMakeQuad(float[] sdf, int[] strideToIndex, List<Vector3> positions, int p1, int p2, int axis_b_stride, int axis_c_stride, List<int> indices)
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
                indices.Add(v1);
                indices.Add(v4);
                indices.Add(v2);

                indices.Add(v1);
                indices.Add(v3);
                indices.Add(v4);
            }
            else
            {
                // [v1, v2, v4, v1, v4, v3]
                indices.Add(v1);
                indices.Add(v2);
                indices.Add(v4);

                indices.Add(v1);
                indices.Add(v4);
                indices.Add(v3);
            }
        }
        else if (negativeFace)
        {
            // [v2, v3, v4, v2, v1, v3]
            indices.Add(v2);
            indices.Add(v3);
            indices.Add(v4);

            indices.Add(v2);
            indices.Add(v1);
            indices.Add(v3);
        }
        else
        {
            // [v2, v4, v3, v2, v3, v1]
            indices.Add(v2);
            indices.Add(v4);
            indices.Add(v3);

            indices.Add(v2);
            indices.Add(v3);
            indices.Add(v1);
        }
    }

    private void GenerateMesh(WorkingData workingData)
    {
        Mesh mesh = new Mesh();
        mesh.vertices = workingData.positions.ToArray();
        mesh.SetIndices(workingData.indices.ToArray(), MeshTopology.Triangles, 0);
        mesh.normals = workingData.normals.ToArray();
        mesh.UploadMeshData(false);
        mesh.RecalculateBounds();

        GetComponent<MeshFilter>().mesh = mesh;
        GetComponent<MeshCollider>().sharedMesh = mesh;
    }

    private void Update()
    {
        if (Input.GetMouseButtonDown(0))
        {
            Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
            if (Physics.Raycast(ray, out RaycastHit hitInfo, float.MaxValue))
            {
                workingData.Reset();

                StartCoroutine(AnimateBlobGrow(hitInfo.point, 0));
            }
        }

        if (Input.GetMouseButtonDown(1))
        {
            Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
            if (Physics.Raycast(ray, out RaycastHit hitInfo, float.MaxValue))
            {
                workingData.Reset();

                StartCoroutine(AnimateBlobGrow(hitInfo.point, 1));
            }
        }

        if (anythingDirty)
        {
            Profiler.BeginSample("MakeAllQuads");
            MakeAllQuads(sdf, new Vector3Int(1, 1, 1), new Vector3Int(CHUNK_SIZE - 2, CHUNK_SIZE - 2, CHUNK_SIZE - 2), workingData);
            Profiler.EndSample();

            Profiler.BeginSample("GenerateMesh");
            GenerateMesh(workingData);
            Profiler.EndSample();

            anythingDirty = false;
        }
    }

    private IEnumerator AnimateBlobGrow(Vector3 pos, int shape)
    {
        Vector3Int posInt = new Vector3Int(Mathf.CeilToInt(pos.x), Mathf.CeilToInt(pos.y), Mathf.CeilToInt(pos.z));
        for (float i = 1.0f; i <= 5f; i += 0.2f)
        {
            Profiler.BeginSample("RenderIntoChunk");

            if (shape == 0)
                RenderSphereIntoChunk(posInt, i, sdf, false);
            else
                RenderSphereIntoChunk(posInt, i, sdf, true);

            Profiler.EndSample();

            Profiler.BeginSample("WorkingData Reset");
            workingData.Reset();
            Profiler.EndSample();

            Profiler.BeginSample("EstimateSurface");
            EstimateSurface(sdf, new Vector3Int(1, 1, 1), new Vector3Int(CHUNK_SIZE - 2, CHUNK_SIZE - 2, CHUNK_SIZE - 2), workingData);
            Profiler.EndSample();

            yield return null;
        }
    }

    void RenderCubeIntoChunk(Vector3Int center, Vector3 size, float[] sdf)
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

        anythingDirty = true;
    }

    void RenderSphereIntoChunk(Vector3Int center, float radius, float[] sdf, bool subtract)
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

                    int lin = Linearize(x, y, z);
                    float curr = sdf[lin];
                    
                    if (subtract)
                        sdf[lin] = Mathf.Max(-val, curr);
                    else
                        sdf[lin] = Mathf.Min(curr, val);

                    dirty[lin] = true;
                }
            }
        }

        anythingDirty = true;
    }

    private void Start()
    {
        anythingDirty = true;

        for (int i = 0; i < CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE; ++i)
        {
            sdf[i] = 10;
            dirty[i] = true;
        }

        for (int i = 0; i < CUBE_CORNERS.Count; ++i)
        {
            LINEARIZED_CUBE_CORNERS[i] = Linearize(CUBE_CORNERS[i]);
        }

        LINEARIZED_XYZ_STRIDES = new int[]
        {
            Linearize(1, 0, 0),
            Linearize(0, 1, 0),
            Linearize(0, 0, 1),
        };

        RenderSphereIntoChunk(new Vector3Int(25, 25, 25), 10f, sdf, false);
        RenderSphereIntoChunk(new Vector3Int(35, 25, 25), 10f, sdf, false);

        workingData.Reset();

        EstimateSurface(sdf, new Vector3Int(1, 1, 1), new Vector3Int(CHUNK_SIZE - 2, CHUNK_SIZE - 2, CHUNK_SIZE - 2), workingData);
        MakeAllQuads(sdf, new Vector3Int(1, 1, 1), new Vector3Int(CHUNK_SIZE - 2, CHUNK_SIZE - 2, CHUNK_SIZE - 2), workingData);
        GenerateMesh(workingData);
    }

    public class WorkingData
    {
        public List<Vector3> positions = new List<Vector3>();
        public List<Vector3> normals = new List<Vector3>();
        public List<int> indices = new List<int>();
        public List<Vector3Int> surfacePoints = new List<Vector3Int>();
        public List<int> surfaceStrides = new List<int>();
        public int[] strideToIndex = null;

        public WorkingData()
        {
            strideToIndex = new int[CHUNK_DATA_LENGTH];

            for (int i = 0; i < CHUNK_DATA_LENGTH; ++i)
            {
                strideToIndex[i] = -1;
            }
        }

        public void Reset()
        {
            Profiler.BeginSample("Clearing Lists");
            positions.Clear();
            normals.Clear();
            indices.Clear();
            surfacePoints.Clear();
            surfaceStrides.Clear();
            Profiler.EndSample();

            Profiler.BeginSample("Resetting StrideToIndex");
            // Doesn't seem like we really need to do this... only "actual" values really get considered.
            /*for (int i = 0; i < CHUNK_DATA_LENGTH; ++i)
            {
                strideToIndex[i] = -1;
            }*/
            Profiler.EndSample();
        }
    }

    private List<Vector3Int> CUBE_CORNERS = new List<Vector3Int>()
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

    private int[] LINEARIZED_CUBE_CORNERS = new int[8];
    private int[] LINEARIZED_XYZ_STRIDES = new int[3];

    private List<Vector2Int> CUBE_EDGES = new List<Vector2Int>()
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

    private List<Vector3> CUBE_CORNER_VECTORS = new List<Vector3>()
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
}
