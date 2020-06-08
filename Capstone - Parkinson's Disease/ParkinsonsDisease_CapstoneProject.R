#****************************************************************************
#...............................LOAD DATASET.................................
#****************************************************************************

#Loading Parkinson's Disease CSV data
pd_data <- read.csv("E:/For_Career/R Projects/My R files/Capstone - Parkinson's Disease/Data/PD_data.csv")
pd_data$status <- as.factor(pd_data$status)
print(pd_data)

#****************************************************************************
#.............................DATA EXPLORATION...............................
#****************************************************************************

#-----------number of rows and columns in the dataset---------------

#no. of rows in pd_data
nrow(pd_data)  #195

#no. of columns in pd_data
ncol(pd_data)   #24

#---------------------------------------------------------------
#column names in Parkinson's Disease Data
colnames(pd_data)

#.....................CHECKING NULL VALUES IN THE DATA.................................

#checking no. of null values in each column
colSums(is.na(pd_data)) #no null values

#---------------------------------------------------------------------------
#checking entries with status 0 and status 1

#checking only 'status' column 
#using a new variable called 'status_val'
status_val<-pd_data[,c("status")] 
print(status_val)

#number of entries with status = 0 i.e. Healthy People
sum(status_val==0) #48

#number of entries with status = 1 i.e. People with Parkinson's Disease
sum(status_val==1) #147
#Total 195

#plotting the ratio of healthy people to people with Parkinson's
barplot(table(pd_data[,18]), xlab = "status (0 = Healthy, 1 = with Parkinsons Disease)", ylab = "No. of patients")

#--------------------------CHECKING REDUNDANCY-------------------------------
#checking for repeated object values in the column "name" in pd_data; redundancy

record_name <- pd_data[,c("name")]
uniq_record_name <- unique(record_name)
length(uniq_record_name)   #195; equal to the number of entries in the data

#Therefore, all the objects in column "name" (i.e. people tested for Parkinson's) and hence their observations for parkinson's are unique.

#---------------------------CHECKING CORRELATION------------------------------
#installing required package
install.packages("dplyr")
library(dplyr)

#removing the name attribute for correlation
pd_data1 <- pd_data[c(2:24)]
pd_data1
colnames(pd_data1)

#creating correlation data
pd_data2 <- transform(pd_data1, status = as.numeric(status))
cor_data <- cor(pd_data2, method = c("spearman"))

#creating correlation matrix
cor_matrix <- round(cor(cor_data),2)

#visualizing correlation matrix
install.packages("corrplot")
library(corrplot)
corrplot::corrplot(cor_matrix, method = "circle")

#------combining correlation values and p-values together in plot for better insight-------

cor.test.mat <- function(mat){
  n <- ncol(mat)
  pmat <- matrix(0, nrow = n, ncol = n)
  for(i in 1:(n-1)){
    for(j in (i+1):n){
      pmat[i,j] <- cor.test(mat[,i], mat[,j], method="pearson")$p.value
    }
  }
  pmat[lower.tri(pmat)] <- t(pmat)[lower.tri(pmat)] #fill lower triangle with upper triangle
  return(pmat)
}  

#compute matrix of p-values
pvals <- cor.test.mat(cor_data)

corrplot::corrplot(cor(cor_data), method="number", order="hclust", addrect=2, diag=F)
corrplot::corrplot(cor(cor_data), p.mat = pvals, sig.level=0, insig = "p-value", method="ellipse", order="hclust", 
         type="upper", addrect=2, tl.pos = "n", cl.pos="n", diag=F, add=T)

#-------using package caret for understanding highly correlated attributes in pd_data-------

install.packages(c("mlbench","caret"))
library(mlbench)
library(caret)

#printing attrbutes that are highly correlated with a cutoff of 0.9
highlyCorrelated <- findCorrelation(cor_matrix, cutoff=0.9)
print(highlyCorrelated)
#The highly correlated attribute no.s are: 23 20  7  4  5 13  6  8 10  9 12 11 14 16  1

#**************************************************************************************
#.........................FEATURE EVALUATION AND SELECTION.............................
#**************************************************************************************

#-------------------------RANKING FEATURES BY IMPORTANCE-----------------------------
#converting list "pd_data1" to data frame
pd_data3 <- as.data.frame(pd_data1)

#fitting a logistic regression model
install.packages("randomForest")
library(randomForest)
feature_model = randomForest(pd_data$status~., data=pd_data3)

#estimate variable importance
importance <- varImp(feature_model)

#summarize importance
print(importance)

#plot importance
varImpPlot(feature_model) 

#Hence, the top 3 attribute features are: PPE, spread 1, MDVP.Fo.Hz

# But other features in this data also play important roles in some way. Therefore, we use PCA to check it out.

#*************************************************************************************
#.........................PRINCIPAL COMPONENT ANALYSIS................................
#*************************************************************************************

#installing packages to apply PCA in Parkinson's Dataset
install.packages("factoextra")
library(factoextra)
library(FactoMineR)

#Doing Principle Component Analysis on the Dataset
pd.pca <- prcomp(pd_data2, center = TRUE, scale = TRUE)
summary(pd.pca)

#2D PCA-plot from 24 feature Parkinson's Disease Dataset
fviz_pca_ind(pd.pca, geom.ind = "point", pointshape = 21, 
             pointsize = 2, 
             fill.ind = pd_data$status, 
             col.ind = "black", 
             palette = "jco", 
             addEllipses = TRUE,
             label = "var",
             col.var = "black",
             repel = TRUE,
             legend.title = "Patient Status") +
  ggtitle("2D PCA-plot from 24 feature dataset") +
  theme(plot.title = element_text(hjust = 0.5))

# Results for Variables
pd.pca.var <- get_pca_var(pd.pca)
pd.pca.var$coord          # Coordinates
pd.pca.var$contrib        # Contributions to the PCs

#Obtaining eigenvalues
pd.eig.val <- get_eigenvalue(pd.pca)
pd.eig.val

#Quality of Representation of variables in PCs
head(pd.pca.var$cos2, 10)
install.packages("ellipsis")
library(ellipsis)
fviz_cos2(pd.pca, choice = "var", axes = 1:3)

#checking quality on the factor map
fviz_pca_var(pd.pca, col.var = "cos2",
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"), 
             repel = TRUE              #Avoid text overlapping
)

#*************************************************************************************
#............................BUILDING PREDICTION MODEL................................
#*************************************************************************************
#Train-Test split
trainIdxs <- sample(x=nrow(pd_data1), size=0.7*nrow(pd_data1), replace=FALSE)
testIdxs <- c(1:nrow(iris))[-trainIdxs]

#------------------------Using CoreModel for classification Prediction--------------------

install.packages("CORElearn")
library(CORElearn)

modelRF <- CoreModel(status~., pd_data1[trainIdxs,], model="rf",
                     selectionEstimator="MDL",minNodeWeightRF=5,
                     rfNoTrees=100, maxThreads=1)

print(modelRF) # simple visualization, test also others with function plot

# prediction on testing set
Y_Pred <- predict(modelRF, pd_data1[testIdxs,], type="both") 

#plotting Y_Pred and Test data for comparison
Y_Test <- as.data.frame(pd_data1[testIdxs,])
Pred <- as.data.frame(Y_Pred$class)
barplot(table(Y_Test[,17]), xlab = "status (0 = healthy, 1 = with Parkinsons)", ylab = "Real_Count")
barplot(table(Pred), xlab = "status (0 = healthy, 1 = with Parkinsons)", ylab = "Prediction_Count")

mEval <- modelEval(modelRF, pd_data1[["status"]][testIdxs], Y_Pred$class, Y_Pred$prob)
print(mEval) # evaluation of the model

#...............................................................................................

# install.packages(c("rmarkdown", "knitr"))
# library("rmarkdown")
# library("knitr")