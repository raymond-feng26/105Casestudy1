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
centroids=all_centroid;
centroid_labels=all_labels;

% loop through the test set, figure out the predicted number
for i = 1:200

testing_vector=test(i,:);

% Extract the centroid that is closest to the test image
[prediction_index, vec_distance]=assign_vector_to_centroid(testing_vector(:,1:784),centroids);

predictions(i) = centroid_labels(prediction_index);

end

%% DESIGN AND IMPLEMENT A STRATEGY TO SET THE outliers VECTOR
% outliers(i) should be set to 1 if the i^th entry is an outlier
% otherwise, outliers(i) should be 0
X_test = test(:,1:784); % 200 rows of iamges * 784 columns of pixel values
C = centroids(:,1:784);
N = size(X_test,1); % number of test images 200
K= size(C,1) % number of representatives (centroids) depending on training


% Squared Euclidean
XX = sum(X_test.^2, 2);                 % N x 1
CC = sum(C.^2, 2)';                     % 1 x K
D2 = XX + CC - 2*(X_test * C');  % N x K
D2 = max(D2, 0);          % makes everything positive, compares the value in Dist_XXCC to zero. 

[minD2, idx_nearest] = min(D2, [], 2); %minD2 is the smallest squared distnace, idx is the index of cloest centroid
distance_nearest = sqrt(minD2);

% threshold: Median + MAD, fallback to 99th percentile if MAD approx 0
medD = median(dist_nearest); %middle value of all distances
MAD  = median(abs(dist_nearest - medD));    % unscaled MAD
% MAD = Median Absolute Deviation 
% = median of the absolute differences from the median. Measures spread
% ignoring extreme outlier. 

if MAD < 1e-9
    s = sort(distance_nearest);
    threshold_idx = max(1, ceil(0.99 * numel(s))); % ceil() rounds up,top 1%
    THRESH = s(threshold_idx);
else
    TAU = 3; % how many MADS above or below median to be considered far
    THRESH = medD + TAU * MAD;
end

outliers = double(dist_nearest > THRESH);
outliers = reshape(outliers, [], 1); % 1 if the ith entry is an outlier


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
sum(correctlabels==predictions)

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

