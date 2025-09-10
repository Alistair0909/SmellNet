# Awesome SmellAI [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
> 🎯 A curated list of papers on smell perception, capture, synthesis, and olfactory interfaces.
>
> 🌸 This repository is the official implementation of **Awesome SmellAI**, an end-to-end pipeline that **captures** odors with an e-nose, **predicts** their composition with a ML model (*ScentRatioNet*), and **releases** reconstructed scents via a 12-channel wearable (Scentrealm Neckwear) over BLE — all in real time.

<div align="center">

<a href="https://github.com/AlistairPernigo" target="_blank">Alistair Pernigo</a> (MIT) • <a href="https://github.com/yunge-wen" target="_blank">Yunge Wen</a> (NYU) • <a href="https://github.com/DeweiFeng" target="_blank">Dewei Feng</a> (MIT) • <a href="https://github.com/ddvd233" target="_blank">David Dai</a> (MIT) • <a href="https://github.com/KaichenZhou" target="_blank">Kaichen Zhou</a> (MIT) • <a href="https://github.com/jasbrooks" target="_blank">Jas Brooks</a> (MIT CSAIL) • <a href="https://github.com/pliang279" target="_blank">Paul Pu Liang</a> (MIT)

<a href="#"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-XXXXX.YYYYY-b31b1b.svg"></a> <a href="#"><img alt="Dataset" src="https://img.shields.io/badge/Dataset-<SmellAI>-green.svg"></a> <a href="https://Alistair0909.github.io/SmellAI/"><img alt="Project Page" src="https://img.shields.io/badge/Project-Website-1f6feb.svg"></a>

</div>

## Gallery
Below we illustrate (i) real-time 4-channel capture, (ii) 12-class mixture prediction, and (iii) BLE release.

<table>
  <tr>
    <td><img src="__assets__/videos/realtime_plot.gif" alt="Realtime Capture"></td>
    <td><img src="__assets__/videos/prediction_bars.gif" alt="12-Class Prediction"></td>
    <td><img src="__assets__/videos/ble_release.gif" alt="BLE Release"></td>
    <td><img src="__assets__/videos/mix_demo.gif" alt="Mixture Demo"></td>
  </tr>
  <tr>
    <td colspan="2"><center>"60 s capture @ 10 Hz from Wio + GMXXX (NO₂ / C₂H₅OH / VOC / CO)"</center></td>
    <td colspan="2"><center>"Release via Scentrealm Neckwear — channels mapped from predicted weights"</center></td>
  </tr>
</table>

Model: **ScentRatioNet** (4-channel time-series → 12-class mixing ratios)

## Steps for Inference

### Prepare Environment
```bash
git clone https://github.com/<YOUR-ORG>/Awesome-SmellAI.git
cd Awesome-SmellAI

# Conda (Windows/Linux/macOS)
conda create -n smellai python=3.10 -y
conda activate smellai

# Python deps
pip install -r requirements.txt
# or (minimal)
# pip install numpy pandas matplotlib joblib pyserial torch torchvision torchaudio bleak crcmod
```

### Download / Place Pretrained Files
```bash
mkdir -p models
# (Copy your .pth and .joblib into the models/ folder)
```

### Generate Weights (capture → predict)
```bash
python scripts/live_capture_to_weights.py \
  --port COM7 --baud 115200 --duration 60 \
  --outdir "data\output" \
  --label "Orange" \
  --model-path "models\mixture_kl_model.pth" \
  --scalers-path "models\channel_scalers_4.joblib" \
  --win-len 600 --hop-len 300 \
  --topk 6 --temperature 1.0 \
  --plot-every 2 --allow-pad
```

This command:
* reads serial CSV (`timestamp_ms,NO2,C2H5OH,VOC,CO`) at 10 Hz,
* shows a **real-time plot**,
* windows and standardizes,
* runs **ScentRatioNet** and saves predictions and device weights.

## Steps for BLE Release (Scentrealm Neckwear)

### Discover & Smoke Test
```bash
python scripts/ble_list_services.py --address 08:F9:E0:E4:57:FE
python scripts/ble_smoke_test.py --address 08:F9:E0:E4:57:FE --channel 8 --seconds 8 --handset 230
```

### Send Predicted Weights (top-k → seconds)
```bash
python scripts/ble_send_weights.py \
  --address 08:F9:E0:E4:57:FE \
  --json "data\output\device_weights_topk.json" \
  --total-seconds 20 --min-per-channel 1 --handset 230
```

## Steps for Training

### Prepare Dataset
* Collect raw streams @10 Hz for each base (12 classes) and mixtures.
* Export to CSV (4 columns). Organize by label.

### Configuration
```yaml
train_data:
  csv_root:        data/train_csv/
  win_len:         600
  hop_len:         300
  normalize:       per_channel
model:
  arch:            ScentRatioNet
  n_channels:      4
  n_classes:       12
opt:
  lr:              3e-4
  batch_size:      32
  epochs:          100
```

### Training
```bash
python train.py --config configs/smellai.yaml
```

## Olfactory Stimulation Approaches in HCI

| Modality | Delivery method | Typical setup/examples | Strengths for HCI | Main limitations | Representative references |
|---|---|---|---|---|---|
| Chemical | Pressurized micro-jets | Provide high-velocity chemical aerosol streams for targeted odor delivery; multiple channels easily combined | Good spatial control and short response time | Requires multiple solenoid valves; limited channel count | [Covington et al. 2018; Dobbelstein et al. 2017] |
| Chemical | Fan-guided airflow | Fast response based on fan guidance; better spatial and temporal precision; low cost | Lower precision and coverage compared to jet streams | [Hartmann 1902; Iwamoto et al. 2009; Matosich et al. 2021] |
| Chemical | Liquid-fueled atomizer | Aerosolizes liquid perfume through a time-of-flight chamber | Better control over channel and plume release; good spatial control | Requires chamber, heating, and chemical cartridges | [Kakehi et al. 2007; Lee et al. 2023; Muñoz-Aguirre et al. 2007; Nakamoto 2016; Nakamoto et al. 2012] |
| Chemical | Ultrasound-driven atomization | Achieves high-speed atomization using ultrasonic vibrations | Excellent temporal control and on-demand release | Complex, limited spatial resolution | [Hasegawa et al. 2018] |
| Chemical | Electronic micro-vaporization | Electronic control for micro vaporizer cartridges | Latent control of scent release | Requires micro vaporizer and control hardware | [Bult et al. 2007; Gerkin 2021] |
| Chemical/Liquid | e-nose | Electronic nose (e-nose) for sensing | — | — | [Conesa Celdrán et al. 2022; Kratz et al. 2022; Lu et al. 2020; Mu et al. 2020] |
| Chemical | Gas-phase matrix | Olfactometer system using mass spectrometry | — | Complex, large, expensive | [Nakamoto 2016; Nakamoto et al. 2012] |
| Chemical | Pressure | Pulsed valve system, high-speed injection | — | — | [Tachimoto 2016] |
| Electrical/Trigeminal | Thermal or chemical stimulation (nose) | Quick reaction; add local nose stimuli to the inhalation cycle | Only stimulates trigeminal nerve; requires precise timing and safety guard | [Brooks et al. 2020; Brooks et al. 2021; Hartmann 1902; Heilig 1962] |

## References
1. Manuel Aleixandre, Dani Prasetyawan, and Takamichi Nakamoto. 2024. Automatic Scent Creation by Cheminformatics Method. *Scientific Reports* 14, 1 (Dec. 2024), 31284. https://doi.org/10.1038/s41598-024-82654-7
2. Judith Amores, Mae Dotan, and Pattie Maes. 2022. Development and Study of Ezzence: A Modular Scent Wearable to Improve Wellbeing in Home Sleep Environments. *Frontiers in Psychology* 13 (March 2022), 791768. https://doi.org/10.3389/fpsyg.2022.791768
3. Judith Amores and Pattie Maes. 2017. Essence: Olfactory Interfaces for Unconscious Influence of Mood and Cognitive Performance. In *Proceedings of the 2017 CHI Conference on Human Factors in Computing Systems*, 28–34. https://doi.org/10.1145/3025453.3026004
4. Virginia Braun and Victoria Clarke. 2006. Using Thematic Analysis in Psychology. *Qualitative Research in Psychology* 3, 2 (Jan. 2006), 77–101. https://doi.org/10.1191/1478088706qp063oa
5. Giada Brianza, Jesse Benjamin, Patricia Cornelio, Emanuela Maggioni, and Marianna Obrist. 2022. QuintEssence: A Probe Study to Explore the Power of Smell on Emotions, Memories, and Body Image in Daily Life. *ACM Transactions on Computer-Human Interaction* 29, 6 (Dec. 2022), 1–33. https://doi.org/10.1145/3526950
6. Jas Brooks, Steven Nagels, and Pedro Lopes. 2020. Trigeminal-Based Temperature Illusions. In *Proceedings of the 2020 CHI Conference on Human Factors in Computing Systems*. 1–12. https://doi.org/10.1145/3313831.3376806
7. Jas Brooks, Shan-Yuan Teng, Jingxuan Wen, Romain Nith, Jun Nishida, and Pedro Lopes. 2021. Stereo-Smell via Electrical Trigeminal Stimulation. In *Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems*. 1–13. https://doi.org/10.1145/3411764.3445300
8. Johannes H.F. Bult, Rene A. de Wijk, and Thomas Hummel. 2007. Investigations on Multimodal Sensory Integration: Texture, Taste, and Ortho- and Retronasal Olfactory Stimuli in Concert. *Neuroscience Letters* 411, 1 (Jan. 2007), 6–10. https://doi.org/10.1016/j.neulet.2006.09.036
9. Molly Burke. 2021. Molly Burke Reviews: Blind Accessibility of Beauty Products.
10. Agustin Conesa Celdrán, Martin John Oates, Carlos Molina Cabrera, Chema Pangua, Javier Tardaguila, and Antonio Ruiz-Canales. 2022. Low-Cost Electronic Nose for Wine Variety Identification through Machine Learning Algorithms. *Agronomy* 12, 11 (Oct. 2022), 2627. https://doi.org/10.3390/agronomy12112627
11. Yulong Chen, Mingjie Li, Wenjun Yan, Xin Zhuang, Kar Wei Ng, and Xing Cheng. 2021. Sensitive and Low-Power Metal Oxide Gas Sensors with a Low-Cost Microelectromechanical Heater. *ACS Omega* 6, 2 (2021), 1216–1222. https://doi.org/10.1021/acsomega.0c05532
12. Eunsol Sol Choi, Yi Xie, Sihao Chen, Elin Carstensdottir, and Edward F. Melcer. 2024. Scented Days: Exploring the Capacity of Smell Narratives. In *CHI PLAY Companion ’24*. 288–293. https://doi.org/10.1145/3665463.3678844
13. James A. Covington, Samuel O. Agbroko, and Akira Tiele. 2018. Development of a Portable, Multichannel Olfactory Display Transducer. *IEEE Sensors Journal* 18, 12 (June 2018), 4969–4974. https://doi.org/10.1109/JSEN.2018.2832284
14. David Dobbelstein, Steffen Herrdum, and Enrico Rukzio. 2017. inScent: A Wearable Olfactory Display as an Amplification for Mobile Notifications. In *Proceedings of the 2017 ACM International Symposium on Wearable Computers*. 130–137. https://doi.org/10.1145/3123021.3123035
15. Andrew Dravnieks. 1982. Odor Quality: Semantically Generated Multidimensional Profiles Are Stable. *Science* 218, 4574 (1982), 799–801. https://doi.org/10.1126/science.7134974
16. Dewei Feng, Carol Li, Wei Dai, and Paul Pu Liang. 2025. SMELLNET: A Large-scale Dataset for Real-world Smell Recognition. https://doi.org/10.48550/ARXIV.2506.00239
17. Idan Frumin, Ofer Perl, Yaara Endevelt-Shapira, Ami Eisen, Neetai Eshel, Iris Heller, Maya Shemesh, Aharon Ravia, Lee Sela, Anat Arzi, and Noam Sobel. 2015. A Social Chemosignaling Function for Human Handshaking. *eLife* 4 (March 2015), e05154. https://doi.org/10.7554/eLife.05154
18. Richard C. Gerkin. 2021. Parsing Sage and Rosemary in Time: The Machine Learning Race to Crack Olfactory Perception. *Chemical Senses* 46 (Jan. 2021), bjab020. https://doi.org/10.1093/chemse/bjab020
19. Sadakichi Hartmann. 1902. A Trip to Japan in Sixteen Minutes.
20. Keisuke Hasegawa, Liwei Qiu, and Hiroyuki Shinoda. 2018. Midair Ultrasound Fragrance Rendering. *IEEE Transactions on Visualization and Computer Graphics* 24, 4 (April 2018), 1477–1485. https://doi.org/10.1109/TVCG.2018.2794118
21. Morton Heilig. 1962. Sensorama Simulator.
22. Rachel S. Herz and Trygg Engen. 1996. Odor Memory: Review and Analysis. *Psychonomic Bulletin & Review* 3, 3 (Sept. 1996), 300–313. https://doi.org/10.3758/BF03210754
23. Yen-Chia Hsu, Jennifer Cross, Paul Dille, Michael Tasota, Beatrice Dias, Randy Sargent, Ting-Hao (Kenneth) Huang, and Illah Nourbakhsh. 2019. Smell Pittsburgh: Community-Empowered Mobile Smell Reporting System. In *Proceedings of the 24th International Conference on Intelligent User Interfaces*. 65–79. https://doi.org/10.1145/3301275.3302293
24. Takuya Iwamoto, Yusuke Sasayama, Mitsuo Motoki, and Takayuki Kosaka. 2009. Back to the Mouth. In *ACM SIGGRAPH 2009 Emerging Technologies*. 1–1. https://doi.org/10.1145/1597956.1597960
25. Yasuaki Kakehi, Motoshi Chikamori, and Kyoko Kunoh. 2007. Hanahana: An Interactive Image System Using Odor Sensors. In *ACM SIGGRAPH 2007 Posters*. 41. https://doi.org/10.1145/1280720.1280766
26. Andreas Keller and Leslie B. Vosshall. 2016. Olfactory Perception of Chemically Diverse Molecules. *BMC Neuroscience* 17, 1 (Dec. 2016), 55. https://doi.org/10.1186/s12868-016-0287-2
27. Sven Kratz, Andrés Monroy-Hernández, and Rajan Vaish. 2022. What’s Cooking? Olfactory Sensing Using Off-the-Shelf Components. In *Adjunct Proceedings of the 35th Annual ACM Symposium on User Interface Software and Technology*. 1–3. https://doi.org/10.1145/3526114.3558687
28. Myron Krueger. 1996. Addition of Olfactory Stimuli to Virtual Reality Simulations for Medical Training Applications. U.S. Army Medical Research and Material Command, Fort Detrick, Frederick, Maryland, USA. 1–110 pages.
29. Christophe Laudamiel. 2025. Osmo Scent Taxonomy: A Perfumer’s Introduction. https://www.generationbyosmo.com/blog/osmo-scent-taxonomy
30. Brian K. Lee, Emily J. Mayhew, Benjamin Sanchez-Lengeling, Jennifer N. Wei, Wesley W. Qian, Kelsie A. Little, Matthew Andres, Britney B. Nguyen, Theresa Moloy, Jacob Yasonik, Jane K. Parker, Richard C. Gerkin, Joel D. Mainland, and Alexander B. Wiltschko. 2023. A Principal Odor Map Unifies Diverse Tasks in Olfactory Perception. *Science* 381, 6661 (Sept. 2023), 999–1006. https://doi.org/10.1126/science.ade4401
31. Junxian Li, Yanan Wang, Zhitong Cui, Jas Brooks, Yifan Yan, Zhengyu Lou, and Yucheng Li. 2025. Mid-Air Gestures for Proactive Olfactory Interactions in Virtual Reality. In *Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems*. 1–18. https://doi.org/10.1145/3706598.3713964
32. Yucheng Li, Yanan Wang, Mengyuan Xiong, Max Chen, Yifan Yan, Junxian Li, Qi Wang, and Preben Hansen. 2025. AromaBite: Augmenting Flavor Experiences Through Edible Retronasal Scent Release. In *Proceedings of the Extended Abstracts of the CHI Conference on Human Factors in Computing Systems*. 1–8. https://doi.org/10.1145/3706599.3720200
33. J. Lozano, M.J. Fernández, J.L. Fontecha, M. Aleixandre, J.P. Santos, I. Sayago, T. Arroyo, J.M. Cabellos, F.J. Gutiérrez, and M.C. Horrillo. 2006. Wine Classification with a Zinc Oxide SAW Sensor Array. *Sensors and Actuators B: Chemical* 120, 1 (Dec. 2006), 166–171. https://doi.org/10.1016/j.snb.2006.02.014
34. Qi Lu, Wan Liang, Hao Wu, Hoiian Wong, Haipeng Mi, and Yingqing Xu. 2020. Exploring Potential Scenarios and Design Implications Through a Camera-like Physical Odor Capture Prototype. In *DIS 2020*. 2021–2033. https://doi.org/10.1145/3357236.3395434
35. Emanuela Maggioni, Robert Cobden, Dmitrijs Dmitrenko, and Marianna Obrist. 2018. Smell-O-Message: Integration of Olfactory Notifications into a Messaging Application to Improve Users’ Performance. In *ICMI ’18*. 45–54. https://doi.org/10.1145/3242969.3242975
36. Daiki Mayumi, Yugo Nakamura, Yuki Matsuda, and Keiichi Yasumoto. 2025. BubblEat: Designing a Bubble-Based Olfactory Delivery for Retronasal Smell in Every Spoonful. In *Proceedings of the Extended Abstracts of the CHI Conference on Human Factors in Computing Systems*. 1–8. https://doi.org/10.1145/3706599.3720047
37. Siddharth Mehrotra, Anke Brocker, Marianna Obrist, and Jan Borchers. 2022. The Scent of Collaboration: Exploring the Effect of Smell on Social Interactions. In *CHI Conference on Human Factors in Computing Systems Extended Abstracts*. 1–7. https://doi.org/10.1145/3491101.3519632
38. Jacquelyn Morie. 2012. The Scent Collar: A Wearable Scent Delivery Device. Institute for Creative Technologies (March 2012).
39. Fanglin Mu, Yu Gu, Jie Zhang, and Lei Zhang. 2020. Milk Source Identification and Milk Quality Estimation Using an Electronic Nose and Machine Learning Techniques. *Sensors* 20, 15 (July 2020), 4238. https://doi.org/10.3390/s20154238
40. Severino Muñoz-Aguirre, Akihito Yoshino, Takamichi Nakamoto, and Toyosaka Moriizumi. 2007. Odor Approximation of Fruit Flavors Using a QCM Odor Sensing System. *Sensors and Actuators B: Chemical* 123, 2 (May 2007), 1101–1106. https://doi.org/10.1016/j.snb.2006.11.025
41. Takamichi Nakamoto. 2016. Olfactory Display and Odor Recorder. In *Essentials of Machine Olfaction and Taste* (1 ed.), Takamichi Nakamoto (Ed.). Wiley, 247–314. https://doi.org/10.1002/9781118768495.ch7
42. T. Nakamoto, M. Ohno, and Y. Nihei. 2012. Odor Approximation Using Mass Spectrometry. *IEEE Sensors Journal* 12, 11 (Nov. 2012), 3225–3231. https://doi.org/10.1109/JSEN.2012.2190506
