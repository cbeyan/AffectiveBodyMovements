%Function to split TSV file into segments as specified in corresponding
%textfile and classify into respective folder

function [] = applyImageGeneration(a, b,func, num,segmentLength,applyRelativePositionExtraction,augmentMode,augmentAngle) % func is the data format, num the file number
    % "a" is the tsv file and "b" is the text file with segments
    % "func" is used to chose the type of data creation and "num" is the index
    % of MoCap data file which is used to label the created images
    % "applyRelativePositionExtraction=1 means; relative position extraction will be applied...
    % 0 means data augmentation will be performed.

    % Herein we give an example for the segment length 4 seconds and it has 0.5 seconds overlap
    % We need to generate images for branch 1 and branch 2
    if segmentLength==4
            segment_overlap_frame=50; % for 2 second segment 0.5 second overlap
            segment_length=400-1; %for 1 second segment
            file_name1 ='Images/FourSecond_05overlap_Branch1/';
            mkdir(file_name1);
            file_name1_2 ='Images/FourSecond_05overlap_Branch2/';
            mkdir(file_name1_2);
    end

    A = importdata(a);
    B = fopen(b,'r');
    sizer = [4 Inf];    
    C = fscanf(B,'%d', sizer)';
    [rs, cs] = size(C);

    for i=1:rs
        key = C(i,cs);  % this is the label: happy, angry, etc...
        data = A(C(i,2):C(i,3), :); % all data has 192 features (64 markers x 3) but they select 30 of them
        required_x = data(:,[1,10,13,28,25,19,91,94,34,40,46,52,190,169,55,64,73,82,187,160,97,106,112,115,121,127,136,142,145,151])';
        required_y = data(:,[2,11,14,29,26,20,92,95,35,41,47,53,191,170,56,65,74,83,188,161,98,107,113,116,122,128,137,143,146,152])';
        required_z = data(:,[3,12,15,30,27,21,93,96,36,42,48,54,192,171,57,66,75,84,189,162,99,108,114,117,123,129,138,144,147,153])';
        count = 1;
    
        [~,total_frames] = size(required_x);
    
        %for j=1:segment_length:total_frames-segment_length % NO OVERLAPPING
        for j=1:segment_overlap_frame:total_frames-segment_length
            a = required_x(:,j:j+segment_length);
            b = required_y(:,j:j+segment_length);
            c = required_z(:,j:j+segment_length);
        
        %if mod(num,2)==0
            allModes=[4 5 6 1]; %-1 stands for no augmentation
            pos = randi(length(allModes)); % randomly generate augmentation
            augmentMode= allModes(pos);
            augmentMode=-1; 
            % AS an example, we set augmentMode=-1, this setting does not apply
            % data augmentation but only create the RGB images belonging to the
            % original 3D Mocap data
            % the path of these images will be
            % ...\Images\FourSecond_05overlap_Branch1\LP_Format\Happy\
            % ...\Images\FourSecond_05overlap_Branch2\LP_Format\Happy\
            % If you want to make random augmentations then set
            % augmentMode=1
            % This will create augmented RGB images and the path of them
            % will be:
            % ...\Images\FourSecond_05overlap_Branch1\AugmentedData\LeftRight\LP_Format\Happy\
            % ...\Images\FourSecond_05overlap_Branch2\AugmentedData\LeftRight\LP_Format\Happy\
            % when the type of the augmentation is determined as L2RR2L
            % swtch
            if augmentMode==-1
                applyRelativePositionExtraction=1;
            else
                applyRelativePositionExtraction=0;
            end
       %else
       %         applyRelativePositionExtraction=1;
       %end
        if segmentLength==4
            segment_overlap_frame=50; % for 2 second segment 0.5 second overlap
            segment_length=400-1; %for 1 second segment
            file_name1 ='Images/FourSecond_05overlap_Branch1/';
            mkdir(file_name1);
            file_name1_2 ='Images/FourSecond_05overlap_Branch2/';
            mkdir(file_name1_2);
        end
        
       if applyRelativePositionExtraction==0 %augmentMode=1; %can be 1, 4, 5 or 6
            if augmentMode==1
                file_name1=[file_name1 'AugmentedData/LeftRight/'];
            elseif augmentMode==4
                file_name1=[file_name1 'AugmentedData/MirrorX/'];
            elseif augmentMode==5
                file_name1=[file_name1 'AugmentedData/MirrorY/'];
            elseif augmentMode==6
                file_name1=[file_name1 'AugmentedData/MirrorZ/'];
            end
        end

            if func==0 % This is for applying logistic function (LP Format)
                    file_name2 =[file_name1 'LP_Format/'];
                    file_name2_2 =[file_name1_2 'LP_Format/'];
                    
                    if applyRelativePositionExtraction==1
                        clav_x = a(5,1);
                        clav_y = b(5,1);
                        clav_z = c(5,1);
            
                        layer1 = a - clav_x;
                        layer2 = b - clav_y;
                        layer3 = c - clav_z;
                    elseif applyRelativePositionExtraction==0
                        if augmentMode==1
                            clav_x = a(5,1);
                            clav_y = b(5,1);
                            clav_z = c(5,1);
            
                            a = a - clav_x;
                            b = b - clav_y;
                            c = c - clav_z;
                        end
                        [a,b,c]=augmentData(a,b,c,augmentMode,augmentAngle);
                        layer1 = a;
                        layer2 = b;
                        layer3 = c;
                    end
                       
                layer1 = (255./(1+exp(-0.0035*(layer1))));
                layer2 = (255./(1+exp(-0.0035*(layer2))));
                layer3 = (255./(1+exp(-0.0035*(layer3))));
            
                pseudo_im = cat(3,round(layer1),round(layer2),round(layer3));
                pseudo_im = uint8(pseudo_im);
                   
                if key == 0
                        mkdir([file_name2 'Angry'])
                        file_name =[file_name2 'Angry/image'];
                        mkdir([file_name2_2 'Angry'])
                        file_name_2 =[file_name2_2 'Angry/image'];
                elseif key == 1   
                        mkdir([file_name2 'Happy'])
                        file_name =[file_name2 'Happy/image'];
                        mkdir([file_name2_2 'Happy'])
                        file_name_2 =[file_name2_2 'Happy/image'];
                elseif key == 2   
                        mkdir([file_name2 'Insecure'])
                        file_name =[file_name2 'Insecure/image'];
                        mkdir([file_name2_2 'Insecure'])
                        file_name_2 =[file_name2_2 'Insecure/image'];
                elseif key == 3
                        mkdir([file_name2 'Sad'])
                        file_name =[file_name2 'Sad/image'];
                        mkdir([file_name2_2 'Sad'])
                        file_name_2 =[file_name2_2 'Sad/image'];
                end
                file_name = [file_name num2str(num) '_sec' num2str(i) '_' num2str(count) '.png'];  
                imwrite(pseudo_im, file_name);
                file_name_2 = [file_name_2 num2str(num) '_sec' num2str(i) '_' num2str(count) '.png'];  
                pseudo_im_2=imcrop(pseudo_im,[round(size(pseudo_im,2)./4.*3)+1,1,round(size(pseudo_im,2)./4),round(size(pseudo_im,1))]);
                imwrite(pseudo_im_2, file_name_2);
                count = count+1;
            end
        end
    end
end