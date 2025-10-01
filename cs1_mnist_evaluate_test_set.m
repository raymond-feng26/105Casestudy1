%% This code evaluates the test set.

% ** Important.  This script requires that:
% 1)'centroid_labels' be established in the workspace
% AND
% 2)'centroids' be established in the workspace
% AND
% 3)'test' be established in the workspace


% IMPORTANT!!:
% You should save 1) and 2) in a file named 'classifierdata.mat' as part of
% your submission.

predictions = zeros(200,1);
outliers = zeros(200,1);
distance_nearest=zeros(200,1); % store all diatances for outlier detection

% loop through the test set, figure out the predicted number
for i = 1:200

testing_vector=test(i,1:784);

% Extract the centroid that is closest to the test image
[prediction_index, vec_distance]=assign_vector_to_centroid(testing_vector,centroids);
distance_nearest(i)=vec_distance;
predictions(i) = centroid_labels(prediction_index);

end


%% DESIGN AND IMPLEMENT A STRATEGY TO SET THE outliers VECTOR
% outliers(i) should be set to 1 if the i^th entry is an outlier
% otherwise, outliers(i) should be 0


% threshold: Median + MAD, fallback to 99th percentile if MAD approx 0
medD = median(distance_nearest); %middle typical value of all distances
MAD  = median(abs(distance_nearest - medD));    % typical deviation
% MAD = Median Absolute Deviation 
% = median of the absolute differences from the median. Measures spread
% ignoring extreme outlier. 

if MAD < 1e-9 % if MAD is small, find the top 1% as outliers
    s = sort(distance_nearest);
    threshold_idx = max(1, ceil(0.99 * numel(s))); % ceil() rounds up,top 1%
    THRESH = s(threshold_idx);
else
    TAU = 2.5; % how many MADS above or below median to be considered far
    THRESH = medD + TAU * MAD;
end

outliers = double(distance_nearest > THRESH); % determine outliers=distance>threshold
outlier_indices = find(outliers == 1); % find outlier index and print out
num_outliers = length(outlier_indices); % number of outliers, for plotting


plotsize = ceil(sqrt(num_outliers));
figure;

for ind = 1:num_outliers
    outlier_sample = test(outlier_indices(ind), 1:784);
    subplot(plotsize, plotsize, ind);
    imagesc(reshape(outlier_sample, [28 28])');
    title(sprintf('Outlier %d', ind));
end

%plot out outliers

%% MAKE A STEM PLOT OF THE OUTLIER FLAG
figure;
stem(1:numel(outliers), outliers); %1:numel(outlier) is the number of outliers in the xaxis, so 1-200
xlabel('Test Set Index');
ylabel('Flag');
title('Outliers');
axis normal;



%% The following plots the correct and incorrect predictions
% Make sure you understand how this plot is constructed
figure;
plot(correctlabels,'o');
hold on;
plot(predictions,'x');
title('Predictions');

%% The following line provides the number of instances where and entry in correctlabel is
% equatl to the corresponding entry in prediction
% However, remember that some of these are outliers
sum(predictions==correctlabels)

function [index, vec_distance] = assign_vector_to_centroid(data, centroids)
    minimumDistance = inf;  % initialize to infinity
    index = 1;  % initialize index
    
    for k = 1:size(centroids, 1)
        distance = norm(data - centroids(k, :));  % calculate distance
        if distance < minimumDistance
            minimumDistance = distance;  % update minimum
            index = k;  % update index
        end
    end
    
    vec_distance = minimumDistance;
end

