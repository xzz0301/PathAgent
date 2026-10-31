# PathoGaze 专家与 PathAgent 同片轨迹 demo

2026-10-09 纠正版已完成：C1/test_001 本地 Agent 原始策略观察5块、最低两轮干预观察8块，两次均 Benign，真值 Malignant。专家最终三例按原始日志和Tobii时钟重建。C1视口6、眼动18；C48视口38、眼动40；C6视口18、眼动16。ThT探索性默认150ms。

主入口：`TECHNICAL_DESIGN.md`。交付：`C:\Users\14910\Documents\ChatGPT\病理医学研究\outputs\PathAgent轨迹对照_周会汇报_20261009\请先阅读_交付索引.md`。

## 运行顺序

在项目根目录按设计文档第11节：pathogaze_reconstruct.py → extract_c1_raw.py → verify_c1_video_alignment.py → reproject_p1_gaze.py → render_corrected_outputs.py。inspect_c1_video.py生成目标录屏抽帧。视频解码使用D盘venv的cv2，其余可用bundled Python。输出 `trajectory_demo\runs\video_audit_20261009`。

通用trajectory_demo.py默认viewport须读取CombinedGazeMouseData.csv或MouseOnlyData.csv，只取浏览操作，不再用GazeOnlyData的fixation采样视口。--point-source gaze为CSV-only探索性；viewport-fixation-legacy仅重现旧版本。最终3例应读取C*_P1_video_aligned_*文件。

视口点表示浏览区域，gaze点表示眼动落点。不能混用。relativeTimestamp不是视频秒数；inImage不是viewer有效性；zoomMagnification不是物镜倍率。dwell使用状态区间，禁止用fixation duration加总冒充视口停留。

## 当前关键输出

- audit_report.json：全库1043902联合行、74258初筛fixation；600浏览任务、37611精简点，50条cap。
- pilot_reprojection_report.json：三例原始对齐与阈值敏感性，分别匹配49/99/39 fixation，有效30/65/28。
- C*_P1_video_aligned_viewport_{full,ThT150}.csv：完整/精简浏览状态。
- C*_P1_video_aligned_gaze_{full,ThT150}.csv和gaze_audit：独立眼动、排除原因、片段来源。
- video_alignment_report.json和验证帧：C1指定视频2:37–2:54的时间/空间证据。
- C1_paired_comparison.json：原始Agent5点对视口6点的原型指标和gaze敏感性。
- 纠正后轨迹图/图片点号核验.json：标签和真实点的对应，不人为移动点。

旧runs/chakraborty_*和viewport_stationary_corrected仅保留历史。旧7/36/15、4/27/11和中间gaze11点均不是最终结果。当前20项专家测试及3项D盘坐标事件测试通过。详细参数、可复现命令和证据边界见设计文档。

Agent运行入口：`D:\PathCode\demo\run_c1_demo.ps1`。默认MinimumRounds=1；-MinimumRounds 2为单独标记的最低覆盖干预。输入在c1_independent，输出按OutputDir分开。SICAP早期跑通数据只作历史，不能与C1配对。
