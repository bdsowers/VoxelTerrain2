using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Profiling;
using Unity.Jobs;
using Unity.Burst;
using Unity.Collections;
using UnityEngine.Rendering;

public class NaiveSurfaceNetsJobified : MonoBehaviour
{
    public readonly static int CHUNK_SIZE = 64;
    public readonly static int CHUNK_DATA_LENGTH = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
    public readonly static Vector3Int CELL_SIZE = new Vector3Int(CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE);

    private NativeArray<float> sdf;
    private NativeArray<Vector3> positions;
    private NativeArray<Vector3> normals;
    private NativeArray<int> indices;
    private NativeArray<int> numIndicies;
    private NativeArray<Vector3Int> surfacePoints;
    private NativeArray<int> surfaceStrides;
    private NativeArray<int> strideToIndex;
    private NativeArray<int> counts;
    private NativeArray<float> cornerDists;
    private NativeArray<Vector3Int> CUBE_CORNERS_NATIVE;
    private NativeArray<Vector2Int> CUBE_EDGES_NATIVE;
    private NativeArray<Vector3> CUBE_CORNER_VECTORS_NATIVE;
    private NativeArray<int> LINEARIZED_XYZ_STRIDES;
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

    NaiveSurfaceNetsJob surfaceDetectionJob;
    MeshGenerationJob meshGenerationJob;
    JobHandle jobHandle;

    public void Start()
    {
        LINEARIZED_XYZ_STRIDES = new NativeArray<int>(3, Allocator.Persistent);
        LINEARIZED_XYZ_STRIDES[0] = Linearize(1, 0, 0);
        LINEARIZED_XYZ_STRIDES[1] = Linearize(0, 1, 0);
        LINEARIZED_XYZ_STRIDES[2] = Linearize(0, 0, 1);

        sdf = new NativeArray<float>(CHUNK_DATA_LENGTH, Allocator.Persistent);
        positions = new NativeArray<Vector3>(CHUNK_DATA_LENGTH, Allocator.Persistent);
        normals = new NativeArray<Vector3>(CHUNK_DATA_LENGTH, Allocator.Persistent);
        dirty = new NativeArray<bool>(CHUNK_DATA_LENGTH, Allocator.Persistent);
        cachedOnSurface = new NativeArray<bool>(CHUNK_DATA_LENGTH, Allocator.Persistent);
        indices = new NativeArray<int>(CHUNK_DATA_LENGTH, Allocator.Persistent);
        numIndicies = new NativeArray<int>(1, Allocator.Persistent);
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

        //SCGDeformations.RenderSphereIntoChunk(new Vector3Int(25, 25, 25), CELL_SIZE, 10f, ref sdf, ref dirty, false);
        //SCGDeformations.RenderSphereIntoChunk(new Vector3Int(35, 25, 25), CELL_SIZE, 10f, ref sdf, ref dirty, false);

        SCGDeformations.RenderCubeIntoChunk(new Vector3Int(CELL_SIZE.x / 2, 2, CELL_SIZE.z / 2), CELL_SIZE, new Vector3(CELL_SIZE.x, 3, CELL_SIZE.z), ref sdf, ref dirty);

        BeginJob();
        CompleteJob();

        Debug.Log("We should see something now...");
    }

    private bool isJobComplete = false;

    private void CompleteJob()
    {
        if (isJobComplete) return;

        jobHandle.Complete();

        Mesh mesh = new Mesh();
        Mesh.ApplyAndDisposeWritableMeshData(meshGenerationJob.meshDataArray, mesh);
        mesh.RecalculateBounds();
        GetComponent<MeshFilter>().mesh = mesh;
        GetComponent<MeshCollider>().sharedMesh = mesh;

        isJobComplete = true;
    }

    private void BeginJob()
    {
        isJobComplete = false;

        surfaceDetectionJob = new NaiveSurfaceNetsJob();
        surfaceDetectionJob.sdf = sdf;
        surfaceDetectionJob.positions = positions;
        surfaceDetectionJob.normals = normals;
        surfaceDetectionJob.surfacePoints = surfacePoints;
        surfaceDetectionJob.surfaceStrides = surfaceStrides;
        surfaceDetectionJob.strideToIndex = strideToIndex;
        surfaceDetectionJob.min = new Vector3Int(1, 1, 1);
        surfaceDetectionJob.max = new Vector3Int(CHUNK_SIZE - 2, CHUNK_SIZE - 2, CHUNK_SIZE - 2);
        surfaceDetectionJob.counts = counts;
        surfaceDetectionJob.cornerDists = cornerDists;
        surfaceDetectionJob.dirty = dirty;
        surfaceDetectionJob.cachedOnSurface = cachedOnSurface;
        surfaceDetectionJob.CELL_SIZE = CELL_SIZE;
        surfaceDetectionJob.CUBE_CORNERS = CUBE_CORNERS_NATIVE;
        surfaceDetectionJob.CUBE_CORNER_VECTORS = CUBE_CORNER_VECTORS_NATIVE;
        surfaceDetectionJob.CUBE_EDGES = CUBE_EDGES_NATIVE;

        meshGenerationJob = new MeshGenerationJob();
        meshGenerationJob.indices = indices;
        meshGenerationJob.min = new Vector3Int(1, 1, 1);
        meshGenerationJob.max = new Vector3Int(CHUNK_SIZE - 2, CHUNK_SIZE - 2, CHUNK_SIZE - 2);
        meshGenerationJob.positions = positions;
        meshGenerationJob.strideToIndex = strideToIndex;
        meshGenerationJob.sdf = sdf;
        meshGenerationJob.surfacePoints = surfacePoints;
        meshGenerationJob.surfaceStrides = surfaceStrides;
        meshGenerationJob.counts = counts;
        meshGenerationJob.LINEARIZED_XYZ_STRIDES = LINEARIZED_XYZ_STRIDES;
        meshGenerationJob.numIndices = numIndicies;
        meshGenerationJob.normals = normals;

        var meshDataArray = Mesh.AllocateWritableMeshData(1);
        meshGenerationJob.meshDataArray = meshDataArray;

        var surfaceDetectionJobHandle = surfaceDetectionJob.Schedule();
        jobHandle = meshGenerationJob.Schedule(surfaceDetectionJobHandle);
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

        if (isSDFDirty && !isJobRunning)
        {
            isSDFDirty = false;

            StartCoroutine(RunFullJob());
        }
    }

    private int Linearize(int x, int y, int z)
    {
        return x + y * CELL_SIZE.y + z * CELL_SIZE.x * CELL_SIZE.y;
    }

    private bool isSDFDirty = false;
    private bool isJobRunning = false;

    private IEnumerator RunFullJob()
    {
        isJobRunning = true;

        BeginJob();
        while (!jobHandle.IsCompleted) yield return null;
        CompleteJob();

        isJobRunning = false;
    }
    private IEnumerator AnimateBlobGrow(Vector3 pos, bool subtract)
    {
        Vector3Int posInt = new Vector3Int(Mathf.CeilToInt(pos.x), Mathf.CeilToInt(pos.y), Mathf.CeilToInt(pos.z));
        for (float i = 1.0f; i <= 5f; i += Time.deltaTime * 15f)
        {
            if (!isJobRunning)
            {
                SCGDeformations.RenderSphereIntoChunk(posInt, CELL_SIZE, i, ref sdf, ref dirty, subtract);
            }

            isSDFDirty = true;

            yield return null;
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

    /*private void GenerateMesh(int numPositions, ref NativeArray<Vector3> positions, int numIndices, ref NativeArray<int> indices, ref NativeArray<Vector3> normals)
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
    }*/

    
}
