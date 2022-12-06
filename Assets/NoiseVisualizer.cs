using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Mathematics;

public class NoiseVisualizer : MonoBehaviour
{
    Texture2D _texture;

    [SerializeField] private NoiseGenerationMethod _generationMethod = NoiseGenerationMethod.Cellular;
    [SerializeField] private Vector3 _period = new Vector3(128, 128, 128);
    [SerializeField] private Vector3 _offset = new Vector3(0, 0, 0);
    [SerializeField] private float _amplitude = 1f;
    [SerializeField] private float _addition = 0f;

    private void Start()
    {
        _texture = new Texture2D(128, 128);
        GetComponentInChildren<UnityEngine.UI.RawImage>().texture = _texture;
    }

    private void Update()
    {
        RenderNoiseToTexture(_texture);
    }

    void RenderNoiseToTexture(Texture2D texture)
    {
        for (int x = 0; x < texture.width; ++x)
        {
            for (int y = 0; y < texture.height; ++y)
            {
                float noise = NoiseHelper.Noise(_generationMethod, x, y, 0, _period, _offset, _amplitude, _addition);
                Color color = new Color(noise, noise, noise);
                texture.SetPixel(x, y, color);
            }
        }

        texture.Apply();
    }
}
