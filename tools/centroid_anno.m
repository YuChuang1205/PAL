data_dir = './masks/';
save_dir = './masks_centroid/';
mkdir(save_dir);
data_list = dir(data_dir);
for i = 3:length(data_list)
    img = imread([data_dir,data_list(i).name]); 
    Ilabel = bwlabel(img); 
    Area_I = regionprops(Ilabel,'centroid');
    img_centroid = zeros(size(img));
    for x = 1: numel(Area_I)
        img_centroid(floor(Area_I(x).Centroid(2)),floor(Area_I(x).Centroid(1))) = 255;
    end
    imwrite(uint8(img_centroid),[save_dir,data_list(i).name])
end