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

test_predictions = zeros(200,1);
outliers = zeros(200,1);
centroids=all_centroid;
centroid_labels=all_labels;

% loop through the test set, figure out the predicted number
for i = 1:200

    votes=zeros(5,1); % use weighted distance voting to find the most confident answer among the 5 models
    distance=zeros(5,1);
    for m=1:5 %for every model
        testing_vector=test(i,:);

        % Extract the centroid that is closest to the test image
        [prediction_index, vec_distance]=assign_vector_to_centroid(testing_vector(:,1:784),centroids{m});
        votes(m)= centroid_labels{m}(prediction_index); % get the vote from each model
        distance(m) = vec_distance; % sstore the distance for weighting, closest the highest
    end
    weight=1./(distance+0.001); % calculate weight
    weightedVotes= accumarray(votes+1, weight,[10,1]); %accumulate array to get weighted votes. 
    %VOtes + 1 convert 0-9 to index 1-10, distance is accumulating data,
    %10,1 is the size
    [~,predictions]=max(weightedVotes);
    test_predictions(i) = predictions-1;
    
end


%% DESIGN AND IMPLEMENT A STRATEGY TO SET THE outliers VECTOR
% outliers(i) should be set to 1 if the i^th entry is an outlier
% otherwise, outliers(i) should be 0

%old------------------------------------------------------------------
%{
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
medD = median(distance_nearest); %middle value of all distances
MAD  = median(abs(distance_nearest - medD));    % unscaled MAD
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

outliers = double(distance_nearest > THRESH);
outliers = reshape(outliers, [], 1); % 1 if the ith entry is an outlier
%}

%old------------------------------------------------------------------

% outlier if: 
% 1. globally far from all centroids
% 2. ambigous between two numbers 

% idea: ensemble of 5 centroids set, for distance and ambiguity checks, treat
% them all together

% Gather ensemble centroids/labels into big matrices
M = numel(centroids); % number of models: 5, 5 random seeds
% Vertically stack all centroids from all models into one big C_total x 784 matrix
C_big = []; % row, holds all centroids stacked together
labels_big = []; % holds all digital label for each centroid in row
for m = 1:M
    C_m = centroids{m}(:,1:784); % centroids for model m
    lbl_m = centroid_labels{m}(:); % labels for model m
    C_big = [C_big; C_m];
    labels_big = [labels_big; lbl_m];% a label per row of C_big, telling which digit that centroid represents
end
C_total = size(C_big, 1); % total number of centroids across all models

% Distances from every test image to every centroid (squared) 
X_test = test(:, 1:784);  % 200 rows of iamges * 784 columns of pixel values
N = size(X_test,1); % number of test images 200

% Squared Euclidean identity 
XX = sum(X_test.^2, 2);                    % N x 1, each test row's ||x||^2
CC = sum(C_big.^2, 2)';                    % 1 x C_total, each centroid's ||c||^2 (row)
D2 = XX + CC - 2*(X_test * C_big');        % N x C_total, all pairwise squared distances
D2 = max(D2, 0);                           % makes everything positive, compares the value in Dist_XXCC to zero. 

% Nearest and 2nd-nearest centroids per test image
[D2_sorted, idx_sorted] = sort(D2, 2, 'ascend'); % sort each row
idx1 = idx_sorted(:,1); % index of nearest centroid
idx2 = idx_sorted(:,2); % index of 2nd nearest centroid
d2_1 = D2_sorted(:,1); % nearest squared distance
d2_2 = D2_sorted(:,2); % 2nd nearest squared distance

% threshold: Median + MAD, fallback to 99th percentile if MAD approx 0
% MAD = Median Absolute Deviation = median of the absolute differences 
% from the median. Measures spread ignoring extreme outlier. 

% Rule 1 to determine outlier: global squared distance
% squared distances directly = robust to outliers and faster.
medD2 = median(d2_1); % median value of nearest squared distance 
MAD2 = median(abs(d2_1 - medD2));% median absolute deviation in sq-dist space

if MAD2 < 1e-12
    % Degenerate spread: mark ~top 1% farthest by percentile
    s2 = sort(d2_1);
    thrshold_idx = max(1, ceil(0.99 * numel(s2))); % ceil() rounds up,top 1%
    THRESH = s2(thr_idx);
else
    TAU = 3.5;  % how many MADS above or below median to be considered far (2.5-3.5)
    THRESH = medD2 + TAU * MAD2;
end
far_flag = d2_1 > THR2; % globally far --> outlier candidate, far_flag is a logical array, T&F


% Rule 2: Ambiguity ("margin") 
% If nearest and 2nd nearest are very close (tie) AND labels differ, flag as ambiguous.
lbl1 = labels_big(idx1); % idx1 = index of the nearest centroid for each test point, label of nearest centroid
lbl2 = labels_big(idx2); % label of 2nd nearest centroid
ratio = d2_1 ./ max(d2_2, eps);  % (0,1], if approach 1 --> more ambiguous, eps prevents divide 0
RATIO_THRESHOLD = 0.95;  % 0-1 means perfect tie, exactly the same position
ambiguous = (ratio > RATIO_THRESHOLD) & (lbl1 ~= lbl2);

% combine outliers, suspicious if one of the rule is violated
trip_count = double(far_flag) + double(ambiguous); % converts logical array to double 1/0
outliers = double(trip_count >= 1); % true when one of the rules in trip_count is true
outliers = outliers(:); % result as colum vector

%check 
sum(far_flag)       % how many flagged by distance 
sum(ambiguous)      % how many flagged by ambiguity
sum(trip_count==2)  % how many flagged by both 

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
plot(test_predictions,'x');
title('Predictions');

%% The following line provides the number of instances where and entry in correctlabel is
% equatl to the corresponding entry in prediction
% However, remember that some of these are outliers
sum(test_predictions==correctlabels)

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

