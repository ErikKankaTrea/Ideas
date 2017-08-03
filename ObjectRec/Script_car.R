
####### IMAGES EXCTRACTED FROM: 
# http://46.100.248.194:8082/view/viewer_index.shtml?id=590
# https://convertio.co/es/jpg-pgm/


# Install the library pixmap to read the pnm files

if(!require("pixmap")){
  install.packages("pixmap")
} else {
  library(pixmap)
}

if(!require("e1071")){
  install.packages("e1071")
} else {
  library(e1071)
}

if(!require("randomForest")){
  install.packages("randomForest")
} else {
  library(randomForest)
}

if(!require("xgboost")){
  install.packages("xgboost")
} else {
  library(xgboost)
}

if(!require("ROCR")){
  install.packages("ROCR")
} else {
  library(ROCR)
}

# Set up our directory
dir <- "C:/Users/Chus/Desktop/car_detection/"
setwd(dir)
# Below the name of positive and negative folders
pos_neg <- c("positive/p_pgm", "negative/n_pgm")



########################## READ POSITIVES 
# Make a loop to load the positive images:
p_files <- list.files(pos_neg[1])

# The matrix to add all the images dimension (num files x ; pixXpix) ---> 21 x 10000
img_pos <- matrix(nrow = length(p_files), ncol = 100*100)

for(i.pfile in 1:length(p_files)){
 
  gray_img <- read.pnm(paste0(pos_neg[1], "/", p_files[i.pfile]))
  img_pos[i.pfile, ] <- c(gray_img@grey)
  
}# end for i.pfile

outcome <- vector(length=length(p_files))
outcome[which(outcome!=1)]=1


########################## READ NEGATIVES
n_files <- list.files(pos_neg[2])
# The matrix to add all the images dimension (num files x ; pixXpix) ---> 21 x 10000
img_neg <- matrix(nrow = length(n_files), ncol = 100*100)

for(i.nfile in 1:length(n_files)){
  
  gray_img <- read.pnm(paste0(pos_neg[2], "/", n_files[i.nfile]))
  img_neg[i.nfile, ] <- c(gray_img@grey)
  
}# end for i.pfile
tmp <- vector(length=length(n_files))
tmp[which(tmp!=0)]=0


# we define the target variable and we add it to our train set:
outcome <- c(outcome, tmp)
data_cars <- data.frame(rbind(img_pos, img_neg))
data_cars$target <- outcome

set.seed(2358)
sample(1:nrow(data_cars), 6) -> filter
test <- data_cars[filter, ]
train <- data_cars[-filter, ]


# Model
forecast_svm = svm(train[, !colnames(train) %in% "target"], train$target)
pred_svm = predict(forecast_svm,test[, !colnames(test) %in% "target"])

forecast_rf = randomForest(train[, !colnames(train) %in% "target"], as.factor(train$target), ntree = 1500)
pred_rf = predict(forecast_rf, test[,!(colnames(test) %in% "target")], type="prob")[,2]


Results<-c()
for(loops in 1:50){
Extreme <- xgboost(data      = data.matrix(train[,!(names(train) %in% "target")]),
                   label       = as.numeric(as.character(train$target)),
                   nrounds     = 9,
                   objective   = "binary:logistic",
                   eval_metric = "auc"
)

Pred_Extreme <-  predict(Extreme, data.matrix(test[,!(names(test) %in% "target")]))
perf_Extreme <- performance (pred_Extreme <- prediction(Pred_Extreme, test[,(names(test) %in% c("target"))]), 'tpr', 'fpr')
gini_Extreme <- (unlist (performance (pred_Extreme, measure = 'auc')@y.values) - .5) / .5

Results[loops]<-gini_Extreme
}

########################## NEW DATA TO CLASIFY ######################
files=list.files('NEW_PICS')
cross= matrix(nrow=NROW(files),ncol=100*100)

for(i in 1:NROW(files))
{
  gray_file=read.pnm(paste('NEW_PICS/',files[i],sep=''))
  cross[i,]=c(gray_file@grey)
}
pred_svm_new_data = predict(forecast_svm, cross, decision.values=TRUE)
pred_rf_new_data = predict(forecast_rf, cross, type="prob")[,2]
pred_xg_new_data = predict(Extreme, cross)

###############copy positives into result directory###############
dir.create('result_rf')
dir.create('result_xg')
dir.create('result_svm')
file.copy(paste('NEW_PICS/', files[which(as.double(pred_svm_new_data)>=0.5)],sep=''),'result_svm/')
file.copy(paste('NEW_PICS/', files[which(as.double(pred_rf_new_data)>=0.5)],sep=''),'result_rf/')
file.copy(paste('NEW_PICS/', files[which(as.double(pred_xg_new_data)>=0.5)],sep=''),'result_xg/')



