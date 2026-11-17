# pairwise-volumetric-fog

山谷里体积雾一开，手电筒光束出现明显断层：froxel 网格按屏幕空间均匀切，深度方向没有做对数分布，近处薄雾和远处浓雾被塞进同一层；光照注入只采样一次，点光源在雾里的散射没有按距离积分；时间重投影直接用上一帧的 froxel，相机移动时雾墙跟着走；半透明物体和雾的排序错了，粒子被雾遮住；方向光阴影贴图和雾的深度切片不对齐，树影边缘出现硬边；HDR 雾颜色没有做 tonemap 前置，亮部过曝。render/volumetric_fog.py 是核心，tests/test_volumetric_fog.py 15个红。要修的是：深度方向对数分布；沿光线积分散射；按 velocity 重投影；透明物体单独合成；阴影切片对齐；在线性空间做雾合成。

## 运行

```bash
python -m pytest tests -q
```

初始状态：15 个用例失败。修复实现使全部用例通过，不要修改 `tests/` 下的测试。

语言/框架：Python 3, pytest；环境可复现等级：无外部依赖。
