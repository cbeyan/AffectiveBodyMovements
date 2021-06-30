function [x_new,y_new,z_new] = augmentData(x,y,z,augmentMode,angle1)

    %x = [1,10,13,28,25,19,91,94,34,40,46,52,190,169,55,64,73,82,187,160,97,106,112,115,121,127,136,142,145,151];
    %y= [2,11,14,29,26,20,92,95,35,41,47,53,191,170,56,65,74,83,188,161,98,107,113,116,122,128,137,143,146,152];
    %z= [3,12,15,30,27,21,93,96,36,42,48,54,192,171,57,66,75,84,189,162,99,108,114,117,123,129,138,144,147,153];
    %Head_Torso: 1=ARIEL 2=C7 3:T5 4:STRN 5=CLAV 6=BWT 7=LBWT 8=RBWT
    %Left_Arm: 9=LSHO 10=LBUPA 11=LELB 12=LIWR 13=LPLM 14=LINDX
    %Right_Arm: 15=RSHO 16=RBUPA 17=RELB 18=RIWR 19=RPLM 20=RINDX
    %Left_Leg: 21=LFTHI 22=LKNI 23=LANK 24=LHEL 25=LMT1
    %Right_Leg: 26=RFTHI 27=RKNI 28=RANK 29=RHEL 30=RMT1

    %mode==1
    %changing left with right markers and vice versa.
    if augmentMode==1
        Left_Arm=[9 10 11 12 13 14];
        Right_Arm=[15 16 17 18 19 20];
        Left_Leg=[21 22 23 24 25];
        Right_Leg=[26 27 28 29 30];
       
        x_new=[x(1:6,:); x(8,:); x(7,:); x(Right_Arm,:); x(Left_Arm,:); x(Right_Leg,:); x(Left_Leg,:)];
        y_new=[y(1:6,:); y(8,:); y(7,:); y(Right_Arm,:); y(Left_Arm,:); y(Right_Leg,:); y(Left_Leg,:)];
        z_new=[z(1:6,:); z(8,:); z(7,:); z(Right_Arm,:); z(Left_Arm,:); z(Right_Leg,:); z(Left_Leg,:)];
    
    elseif augmentMode==2
        %mode==2
        %2D rotation
        %first apply 'relative position extraction'. 
        clav_x = x(5,1);
        clav_y = y(5,1);
        clav_z = x(5,1);
        x = x - clav_x;
        y = y - clav_y;
        z = z - clav_z;
        %2D rotation is, when marker is (x,y)
        %x' = x cos ? ? y sin ?
        %y' = x sin ? + y cos ?
        for i=1:size(x,1) %30
                x_new(i,:)=x(i,:).*cos(angle1)+y(i,:).*sin(angle1);
                y_new(i,:)=x(i,:).*sin(angle1)+y(i,:).*cos(angle1);
        end
        z_new=z;
    elseif augmentMode==3
        %mode==3
        %3D rotation
        clav_x = x(5,1);
        clav_y = y(5,1);
        clav_z = x(5,1);
        x = x - clav_x;
        y = y - clav_y;
        z = z - clav_z;
        %3D rotation is, when marker is (x,y,z)
        u = [1 1 1]';
        for i=1:size(x,1) %30
                %V=[x(i,:)';y(i,:)';z(i,:)'];
                V=[x(i,:); y(i,:); z(i,:)];
                R=rotate_3D(V, 'any', angle1, u);
                x_new(i,:)=R(1,:)';
                y_new(i,:)=R(2,:)';
                z_new(i,:)=R(3,:)'; 
                

        end
 
    elseif augmentMode==4 % 3D mirror on X used for visualization
        clav_x = x(5,1);
        clav_y = y(5,1);
        clav_z = x(5,1);
        x = x - clav_x;
        y = y - clav_y;
        z = z - clav_z;
        %3D rotation is, when marker is (x,y,z)
        u = [1 0 0]';
        for i=1:size(x,1) %30
                %V=[x(i,:)';y(i,:)';z(i,:)'];
                V=[x(i,:); y(i,:); z(i,:)];
                R=rotate_3D(V, 'any', angle1, u);
                x_new(i,:)=R(1,:)';
                y_new(i,:)=R(2,:)';
                z_new(i,:)=R(3,:)'; 
                

        end
        
    elseif augmentMode==5 % 3D mirror on Y used for visualization
        clav_x = x(5,1);
        clav_y = y(5,1);
        clav_z = x(5,1);
        x = x - clav_x;
        y = y - clav_y;
        z = z - clav_z;
        %3D rotation is, when marker is (x,y,z)
        u = [0 1 0]';
        for i=1:size(x,1) %30
                %V=[x(i,:)';y(i,:)';z(i,:)'];
                V=[x(i,:); y(i,:); z(i,:)];
                R=rotate_3D(V, 'any', angle1, u);
                x_new(i,:)=R(1,:)';
                y_new(i,:)=R(2,:)';
                z_new(i,:)=R(3,:)'; 
                

        end
        
    elseif augmentMode==6 % 3D mirror on Z used for visualization
        clav_x = x(5,1);
        clav_y = y(5,1);
        clav_z = x(5,1);
        x = x - clav_x;
        y = y - clav_y;
        z = z - clav_z;
        %3D rotation is, when marker is (x,y,z)
        u = [0 0 1]';
        for i=1:size(x,1) %30
                %V=[x(i,:)';y(i,:)';z(i,:)'];
                V=[x(i,:); y(i,:); z(i,:)];
                R=rotate_3D(V, 'any', angle1, u);
                x_new(i,:)=R(1,:)';
                y_new(i,:)=R(2,:)';
                z_new(i,:)=R(3,:)'; 
                

        end
    elseif augmentMode==10 % used for visualization
        clav_x = x(5,1);
        clav_y = y(5,1);
        clav_z = z(5,1);
        x_new = x - clav_x;
        y_new = y - clav_y;
        z_new = z - clav_z;     
        
    end
end