using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Mathematics;

public static class NoiseHelper
{
    public static float Noise(NoiseGenerationMethod generationMethod, float x, float y, float z, Vector3 period, Vector3 offset, float amplitude, float addition)
    {
        return Noise(generationMethod, x / period.x + offset.x, y / period.y + offset.y, 1 / period.z + offset.z) * amplitude + addition;
    }

    public static float Noise(NoiseGenerationMethod method, float x, float y, float z)
    {
        if (method == NoiseGenerationMethod.Cellular)
        {
            return noise.cellular(new float3(x, y, z)).x;
        }
        else if (method == NoiseGenerationMethod.CNoise)
        {
            return noise.cnoise(new float3(x, y, z));
        }
        else if (method == NoiseGenerationMethod.PNoise)
        {
            return noise.pnoise(new float3(x, y, z), new float3(64, 64, 64));
        }
        else if (method == NoiseGenerationMethod.SNoise)
        {
            return noise.snoise(new float3(x, y, z));
        }

        return 0f;
    }
}

public enum NoiseGenerationMethod
{
    Cellular,
    CNoise,
    PNoise,
    SNoise,
}
