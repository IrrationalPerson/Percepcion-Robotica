# Automated Student Detection for Safety Assurance within Challenge-Based Learning
\t  Detecting the pose of an student in an educational workspace that has heavy machinery or dangerous hazards is a safety precaution that in automated safeguard protocols is not yet applied. To address this gap, we propose a novel solution that combines the Intel® RealSense™ Depth Camera D435 with the YOLOv8 algorithm for object detection. The utilization of **YOLOv8**, is a technology which allows for rapid detection of objects of interest, making it ideal for maintaining low response times. 

  Additionally, the integration of stereo camera technology enables the precise identification of human positions within an _XY_ divided space. The convolutional neural network architecture formulated by the algorithm enables the ability to identify and classify human beings on an image with a confidence level superior to 50%. Once this is done, the infrared sensors in the camera are utilized to deliver an estimate of the distance in meters from the camera to the median location of the person in the picture. 

  With the use of triangulation performed by the stereoscopic vision with a reference marker, one can proceed to generate an approximation of the individual's relative coordinates in the closed space where the task is monitored. This technology empowers students to proactively assess their proximity to restricted or hazardous areas in educational laboratories, enhancing their safety awareness. By avoiding potentially dangerous spaces, students increase their adherence to safety protocols, leading to improved work efficiency and productivity within real workshop settings.


## Research Project Structure

  The work presented in this paper delineates the integration of an object detection algorithm in the eighth version of YOLO alongside the  Intel® RealSense™ Depth Camera D435 to capture the depth measurements of one or multiple students in a laboratory or workshop containing heavy, potentially perilous machinery. Once their distance to the camera is determined, through triangulation of a reference marker, this being a STOP sign, and some coordinate space transformations, one can obtain the relative location of the persons in the workspace with reference to the security sign. 

  If the students fail to respect the distance guidelines for the lab, a security protocol is activated to indicate them and the lab supervisors of the situation to proceed to correct it and thus influence the mentality of the students towards good safety practices. The results point out to an increase in lab productivity under safety guidelines for the students at a higher education level thanks to this interactive form of protocol learning.

![Research Structure](https://github.com/IrrationalPerson/Percepcion-Robotica/assets/21211365/7f758d58-bbda-474c-a477-94afbae1022b)


## Program Structure

  The first part of the code was dedicated to set the initial configurations of either the camera or the algorithm. The first step involved working directly with the **YOLOv8** algorithm and with the already trained neural network that used the **COCO** dataset. 

  The following step of the integration was deciding which object would represent our origin. As the STOP sign was already a part of the classes provided by the trained model, given its significance as warning symbol, it will be the reference object to be  detected as the origin of the workspace. The only two classes that are going to be used for the main code will then be the STOP sign (class number 11) and a person (class number 0).

  The next step was to establish communication with the camera and get the intrinsic properties of it to use them later for calculating distances. 

  The second part of the code, was related to the object pose detection and coordinates calculation. By using the **YOLOv8** algorithm, the program first searches for the STOP sign inside the workspace. When detected, the program saves the coordinates and sets the _Danger_ and the _Safe Zone_ for the Red and Green Protocols respectively. 

  After the origin is set, the program will be continuously searching for persons inside the workspace and reproduce the image taken of the workspace. If a person is detected, the program will calculate the median of the bounding box containing the person and then get the pose with respect to the origin. Afterwards, the coordinates are rotated with a transformation matrix to align the position of the student with the axes plane of the machine that is being monitored.

  The final step involves analyzing the position of the person in either in the _Danger_ or _Safe Zone_. Activation of the _Red Protocol_ occurs if the person is in the _Danger Zone_, changing the bounding box color to red and triggering a flag for protocol activation. This flag can initiate safety measures, such as warning sounds, a speaker with a message or anything that ensures the safety of the operator. On the other hand, if the person is in the _Safe_ zone, the _Green Protocol_ is activated, the bounding box changes color to green and a corresponding flag for safety measures is triggered. 

![Program Structure](https://github.com/IrrationalPerson/Percepcion-Robotica/assets/21211365/120d5080-34ae-47f0-b485-315dc37f1d5a)


## References

[1] A. Villanueva, “El Tec transforma su modelo educativo; será más flexible y vivencial,” _CONECTA_, 2018.

[2] T. S. Love, K. R. Roy, M. Gill, and M. Harrell, “Examining theinfluence that safety training format has on educators’ perceptions of safer practices in makerspaces and integrated stem labs,” _Journal of safety research_, vol. 82, pp. 112–123, 2022.

[3] C. Zhou, D. Ren, X. Zhang, C. Yu, and L. Ju, “Human positiondetection based on depth camera image information in mechanicals afety,” _Advances in Mathematical Physics_, vol. 2022, pp. 1–10, 2022.

[4] S. Secil and M. Ozkan, “Minimum distance calculation using skeletal tracking for safe human-robot interaction,” _Robotics and Computer-Integrated Manufacturing_, vol. 73, p. 102253, 2022.

[5] J. Berger and S. Lu, “A multi-camera system for human detection and activity recognition,” _Procedia CIRP_, vol. 112, pp. 191–196, 2022.

[6] M. Younsi, M. Diaf, and P. Siarry, “Automatic multiple moving humans detection and tracking in image sequences taken from a stationary thermal infrared camera,” _Expert Systems with Applications_, vol. 146,p. 113171, 2020.

[7] B. Gaikwad and A. Karmakar, “Smart surveillance system for real-time multi-person multi-camera tracking at the edge,” _Journal of Real-Time Image Processing_, vol. 18, no. 6, pp. 1993–2007, 2021.

[8] A. Sharma, S. Anand, and S. K. Kaul, “Intelligent querying for target tracking in camera networks using deep q-learning with n-step boot-strapping,” _Image and Vision Computing_, vol. 103, p. 104022, 2020.

[9] N. H. Abdulghafoor and H. N. Abdullah, “A novel real-time multiple objects detection and tracking framework for different challenges,” _Alexandria Engineering Journal_, vol. 61, no. 12, pp. 9637–9647, 2022.

[10] J. Liu, X. Chen, C. Wang, G. Zhang, and R. Song, “A person-following method based on monocular camera for quadruped robots,” _Biomimetic Intelligence and Robotics_, vol. 2, no. 3, p. 100058, 2022.

[11] D. Lim, J. Kim, and H. Kim, “Efficient robot tracking system using single-image-based object detection and position estimation,” _ICT Ex-press_, 2023.

[12] D. Reis, J. Kupec, J. Hong, and A. Daoudi, “Real-time flying object detection with YOLOv8,” _arXiv preprint_ arXiv:2305.09972, 2023.

[13] J. Redmon and A. Farhadi, “YOLOv3: An incremental improvement,” _arXiv preprint_ arXiv:1804.02767, 2018.

[14] C. X. Ge, M. A. As’ari, and N. A. J. Sufri, “Multiple face mask wearer detection based on yolov3 approach,” _IAES International Journal of Artificial Intelligence_, vol. 12, no. 1, p. 384, 2023.

[15] A. U. Rehman, Y. Khan, R. U. Ahmed, N. Ullah, and M. A. Butt, “Human tracking robotic camera based on image processing for live streaming of conferences and seminars,” _Heliyon_, vol. 9, no. 8, 2023.

[16] M. Labussière, C. Teuli`ere, and O. Ait-Aider, “Blur aware metric depth estimation with multi-focus plenoptic cameras,” _Computer Vision and Image Understanding_, vol. 235, p. 103802, 2023.

[17] H. Bozorgi, X. T. Truong, H. M. La, and T. D. Ngo, “2d laser and 3d camera data integration and filtering for human trajectory tracking,” in _2021 IEEE/SICE International Symposium on System Integration (SII)_ ,pp. 634–639, IEEE, 2021.

[18] E. Petrović, A. Leu, D. Ristić-Durrant, and V. Nikolić, “Stereo vision-based human tracking for robotic follower,” _International Journal of Advanced Robotic Systems_, vol. 10, no. 5, p. 230, 2013.

[19] D. Gorodnichy, S. Malik, and G. Roth, “Affordable 3d face tracking using projective vision,” in _Proc. of Int. Conf. on Vision Interface_, pp. 383–390, 2002.

[20] M. Hussein, W. Abd-Almageed, Y. Ran, and L. Davis, “Real-time human detection, tracking, and verification in uncontrolled camera motion environments,” in _Fourth IEEE International Conference on Computer Vision Systems (ICVS’06)_, pp. 41–41, IEEE, 2006.

[21] T. Alanazi, K. Babutain, and G. Muhammad, “A robust and automated vision-based human fall detection system using 3d multi-stream cnn swith an image fusion technique,” _Applied Sciences_, vol. 13, no. 12,p. 6916, 2023.

[22] Y. Xiao, V. R. Kamat, and C. C. Menassa, “Human tracking from single rgb-d camera using online learning,” _Image and Vision Computing_, vol. 88, pp. 67–75, 2019.

[23] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in _Proceedings of the IEEE conference on computer vision and pattern recognition_, pp. 770–778, 2016.

[24] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez,Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” _Advances in neural information processing systems_, vol. 30, 2017.

[25] Intel, Intel® RealSenseTM Product Family D400 Series, 9 2023. Rev.017.

[26] S. Dougherty, J. R. Simpson, R. R. Hill, J. J. Pignatiello, and E. D. White, “Effect of heredity and sparsity on second-order screening design performance,” _Quality and Reliability Engineering International_, vol. 31, no. 3, pp. 355–368, 2015.
