data_dir = './masks/';
save_dir = './masks_coarse/';
mkdir(save_dir);
data_list = dir(data_dir);
for i = 3:length(data_list)
    img = double(imread([data_dir,data_list(i).name])); 
    Ilabel = bwlabel(img); 
    BoundingBoxs = regionprops(Ilabel,'BoundingBox');
    centroids = regionprops(Ilabel,'centroid');
    img_centroid = zeros(size(img));
    for j = 1: numel(BoundingBoxs)
        img_temp = zeros(size(img));
        while sum(sum(img_temp .* img)) == 0
            gaussian_num_x = normrnd(0,1/4,1,1) ;
            gaussian_num_y = normrnd(0,1/4,1,1) ;
            x = floor(centroids(j).Centroid(1)+BoundingBoxs(j).BoundingBox(3)/2*gaussian_num_x);
            y = floor(centroids(j).Centroid(2)+BoundingBoxs(j).BoundingBox(4)/2*gaussian_num_y);
            [y_max, x_max] = size(img);
            if x <1
                x= 1;
            end
            if y <1
                y= 1;   
            end
            if x>x_max
                x = x_max;
            end
            if y>y_max
                y = y_max;
            end
            img_temp(y,x) = 255;
        end
        img_centroid(y,x) = 255;
    end
    imwrite(uint8(img_centroid),[save_dir,data_list(i).name])
end

function [y] = Gaussian(x,mu,sigma)
y = exp(-(x-mu).^2/(2*sigma^2));%
end