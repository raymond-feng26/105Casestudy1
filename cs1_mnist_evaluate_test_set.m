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
all_distance=zeros(200,10);
centroids=all_centroid;
centroid_labels=all_labels;

% loop through the test set, figure out the predicted number
for i = 1:200

    votes=zeros(10,1); % use weighted distance voting to find the most confident answer among the 10 models
    distance=zeros(10,1);
    for m=1:10 %for every model
        testing_vector=test(i,:);

        % Extract the centroid that is closest to the test image
        [prediction_index, vec_distance]=assign_vector_to_centroid(testing_vector(:,1:784),centroids{m});
        votes(m)= centroid_labels{m}(prediction_index); % get the vote from each model
        distance(m) = vec_distance; % sstore the distance for weighting, closest the highest
    end
    
    all_distance(i,:)=distance; % store distance for outlier detection
    weight=1./(distance+0.001); % calculate weight
    weightedVotes= accumarray(votes+1, weight,[10,1]); %accumulate array to get weighted votes. 
    %VOtes + 1 convert 0-9 to index 1-10, distance is accumulating data,
    %10,1 is the size
    [~,predictions]=max(weightedVotes); % find prediction as maximum vote
    test_predictions(i) = predictions-1; % convert from 1-10 to 0-9
    
end


%% DESIGN AND IMPLEMENT A STRATEGY TO SET THE outliers VECTOR
% outliers(i) should be set to 1 if the i^th entry is an outlier
% otherwise, outliers(i) should be 0


distance_nearest = min(all_distance, [], 2); %200*1 vector of smallest distance to centroid

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
    TAU = 3; % how many MADS above or below median to be considered far
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

