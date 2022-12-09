using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Jobs;
using Unity.Burst;
using Unity.Collections;
using UnityEngine.Rendering;

public struct BakePhysicsMeshJob : IJob
{
    public int meshId;

    public void Execute()
    {
        Physics.BakeMesh(meshId, false);
    }
}
