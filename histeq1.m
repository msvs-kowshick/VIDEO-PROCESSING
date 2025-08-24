clc;  
% Step 1: Read the Video  
videoFile = 'D:\Downloads\sub-2.mp4'; % Input noisy video file  
outputFile = 'D:\Downloads\enhanced_video2.mp4'; % Output video file  
denoisedOutputFile = 'D:\Downloads\denoised_video2.mp4'; % Denoised video output  
superResOutputFile = 'D:\Downloads\super_resolution_video2.mp4'; % Super-resolved video output  

vidObj = VideoReader(videoFile);  
numFrames = round(vidObj.Duration * vidObj.FrameRate);  
frameWidth = vidObj.Width;  
frameHeight = vidObj.Height;  

% Step 2: Create VideoWriter Objects  
outputVid = VideoWriter(outputFile, 'MPEG-4');  
open(outputVid);  

denoisedVid = VideoWriter(denoisedOutputFile, 'MPEG-4');  
open(denoisedVid);  

superResVid = VideoWriter(superResOutputFile, 'MPEG-4');  
open(superResVid);  

% Initialize optical flow object and motion vector arrays  
opticFlow = opticalFlowLK('NoiseThreshold', 0.01);  
cumulativeMotionX = 0;  % Cumulative motion in X direction  
cumulativeMotionY = 0;  % Cumulative motion in Y direction  

% Step 3: Process Each Frame  
for k = 1:numFrames  
    % Read a frame  
    frame = read(vidObj, k);  
    
    % Convert to grayscale for optical flow  
    grayFrame = rgb2gray(frame);  
    
    % Stabilization: Calculate optical flow between frames (except the first frame)  
    if k > 1  
        flow = estimateFlow(opticFlow, grayFrame);  
        cumulativeMotionX = cumulativeMotionX + mean(flow.Vx(:)); % Average motion in X  
        cumulativeMotionY = cumulativeMotionY + mean(flow.Vy(:)); % Average motion in Y  
        
        % Create a transformation to counter the motion  
        tform = affine2d([1 0 0; 0 1 0; -cumulativeMotionX -1*(cumulativeMotionY) 1]);  
        
        % Create an image reference object matching the original frame size  
        outputRef = imref2d(size(frame));  
        
        % Stabilize frame using imwarp with the transformation and the reference  
        stabilizedFrame = imwarp(frame, tform, 'OutputView', outputRef);  
    else  
        stabilizedFrame = frame; % First frame, no stabilization needed  
    end  
    
    % Step 4: Apply Non-Local Means Denoising  
    denoisedFrame = imnlmfilt(stabilizedFrame, 'DegreeOfSmoothing', 0.01);  

    % Save denoised frame to denoised video  
    writeVideo(denoisedVid, uint8(denoisedFrame));  
    
    % Step 5: Enhance Resolution (Super Resolution)  
    superResFrame = imresize(denoisedFrame, 2, 'bilinear'); % Increase size by a factor of 2  
    
    % Save super-resolved frame to super-resolved video  
    writeVideo(superResVid, uint8(superResFrame));  
    
    % Step 6: Enhance Contrast using histogram equalization  
    enhancedFrame = histeq(uint8(superResFrame)); % Enhance contrast  
    
    % Write the processed frame to the output video  
    writeVideo(outputVid, uint8(enhancedFrame));  
end  

% Step 7: Close VideoWriter objects  
close(outputVid);  
close(denoisedVid);  
close(superResVid);  
disp('Video processing complete!');