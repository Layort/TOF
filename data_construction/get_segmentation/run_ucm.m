% MATLAB script to get Ultrametric Contour Maps for images:
% Clone this github repo first:
% https://github.com/jponttuset/mcg/tree/master/pre-trained
%
% Author: Ankush Gupta

% path to the directory containing images, which need to be segmented
img_dir = '../ori_img/';
% path to the mcg/pre-trained directory.
mcg_dir = './pre-trained';
fprintf('%s\n',img_dir)
im = [240,NaN];
% "install" the MCG toolbox:
run(fullfile(mcg_dir,'install.m'));
% get the image names:
imname = dir(fullfile(img_dir,'*'));
imname = imname(arrayfun(@(x) ~strcmp(x.name, '.') && ~strcmp(x.name, '..'), imname));

imname = {imname.name};
length(imname)
%delete(gcp('nocreate')) %关闭并行计算池

% process:
names = cell(numel(imname),1);
ucms = cell(numel(imname),1);

% 使用 parfor 循环并行处理当前批次的图像
% parpool('local',10);
parfor i = 1:numel(imname)
	fprintf('%d of %d\n',i,numel(imname));
	try
    	im_name = fullfile(img_dir,imname{i});
		im = imread(im_name);
	catch
		fprintf('err\n');
		continue;
    end 
		im = uint8(imresize(im,imsize));
		names{i} = imname{i};
		ucms{i} = im2ucm(im,'fast');
end
save('ucm.mat','ucms','names','-v7.3');
fprintf('over')