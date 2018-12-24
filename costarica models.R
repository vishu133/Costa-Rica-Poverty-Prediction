library(rattle)
library(caret)
library(rpart)
EnsurePackage <- function(x) {
  
  x <- as.character(x)
  if(!require(x,character.only = T))
  {install.packages(x,pkgs = "https://cran.cnr.berkeley.edu")}
  require(x,character.only = T)
  
}

EnsurePackage("shiny") #app
EnsurePackage("ggplot2")#data visualization
EnsurePackage("dplyr") #selecting data
EnsurePackage("magrittr") #using pipe operators
EnsurePackage("reshape2") #data processing
EnsurePackage("caret") #data wrangling
EnsurePackage("mlbench") #machine learning
EnsurePackage("e1071") #naive bayes
EnsurePackage("plyr") #manipulation
EnsurePackage("corrplot") #correlation
EnsurePackage("randomForest") #randomforest
EnsurePackage("Hmisc") #correlation 
EnsurePackage("ROCR") #create roc curve
EnsurePackage("ElemStatLearn")
EnsurePackage("arules") #discretization
EnsurePackage("factoextra") #pca visualisation

train <- read.csv("train.csv",header = T,stringsAsFactors = F)

full <- as_tibble(train)


#find NA across all

full_missing = full[, sapply(full, anyNA), drop = FALSE]

cat("Missing data found in ", ncol(full_missing)/ncol(full)*100, "% of features")


op <- par(mar = c(5,8,4,2) + 0.1)
aggr <- lapply(full_missing, function(x) {sum(is.na(x))})
df <- melt(aggr)
barplot(height = df$value,names.arg = c("Mnthly Rnt Paymnt",
                                        "No of Tablet Household owns",
                                        "Yrs behind School",
                                        "avg yrs of education",
                                        "Square of mean years"),las=2,
        horiz = T,xlim=c(0,10000),cex.names = 0.8,col="red")
par(op)

aggr


# The missing columns in the dataset are 
# v2a1:monthly rent
# v18q1:number of tablets owned put this as 0
# rez_esc:years behind in school put this as 0
# meaneduc:mean years of education
# SQBmeaned: square of the mean years of education of adults


d <- density(train$v2a1,na.rm = T)
plot(d, main="Monthly Rent Payment",xlim=c(0,1000000),xlab = "Amount")
polygon(d, col="red", border="blue")


#Have an assumption that those whos rent is NA own the house we have a variable tipovivi for that!

#All variables already one hot encoded
#convert all one hot encoded variables into single features

feature_list = c(
  "pared",
  "piso",
  "techo",
  "abasta",
  "sanitario",
  "energcocinar",
  "elimbasu",
  "epared",
  "etecho",
  "eviv",
  "estadocivil",
  "parentesco",
  "instlevel",
  "tipovivi",
  "lugar",
  "area"
)

#Matrix to store new feature
new_features_integer = data.frame(matrix(ncol = length(feature_list), nrow = nrow(full)))
ohe_names = vector()

for(i in 1:length(feature_list)){
  
  # Grab the feature
  
  feature.to.fix = train %>% select(starts_with(feature_list[i]))
  
  # Fix and enter into our new feature matrix
  
  new_features_integer[,i] = as.integer(factor(names(feature.to.fix)[max.col(feature.to.fix)], 
                                               ordered = FALSE))
  names(new_features_integer)[i] = paste0(feature_list[i],"_int")
  
  ohe_names = c(ohe_names, as.vector(names(feature.to.fix)))
  
}

full <- data.frame(cbind(train,new_features_integer))


#Checking tipovivi_int for houses with NA rent
#Some data exploration on those whos rent is NA

zerorentdataset <- full[is.na(train$v2a1),]
table(zerorentdataset$Target,zerorentdataset$tipovivi_int)

#We see that lots of people who own house have NA rent therefore we 
#will assign value 0 to all NA v2a1
full[is.na(full[,"v2a1"]),"v2a1"] <- 0

#Not everybody owns a tablet will assign 0 to all NA tabs
full[is.na(full[,"v18q1"]),"v18q1"] <- 0

#Years behind school will also be assumed as 0 for NA
full[is.na(full[,"rez_esc"]),"rez_esc"] <- 0

#checking nature of missing educmean values
full[is.na(full[,"meaneduc"]),"Target"]

#Missing values in mean education will be replaced with median
#since all missing values have target value 4
full[is.na(full[,"meaneduc"]),"meaneduc"] <- median(full$meaneduc,
                                                    na.rm = T)

#Same goes for sqaured mean
full[is.na(full[,"SQBmeaned"]),"SQBmeaned"] <- median(full$SQBmeaned,
                                                    na.rm = T)

#converting multinomial to binomial
full$Target <- ifelse(full$Target == 4,0,1)

#Removing columns with little use and columns containing only one value
full <- full[,sapply(full,function(x) nlevels(as.factor(x))>1)]
full <- subset(full,select= -c(Id,idhogar))


#trainf converts all columns in factor good for naive bayes
fullf <- full %>% mutate_all(as.factor)

#Creating a tibble with only numeric values
fulln <-full %>% select_if(is.numeric)

#Converting target into label
full$Target <- as.factor(full$Target)

#Meaneduc has lot of categories will cause problem in classification
#discretizing the values into categories

fullf$meaneduc <- as.numeric(fullf$meaneduc)
fullf$meaneduc <- discretize(fullf$meaneduc,
                            method = "fixed",
                            breaks = c(-Inf,0,2,4,6,8,10,
                                       12,14,18,20,Inf),
                          labels = c("0","1","2",
                                    "3","4","5","6","7",
                                    "8","9","10"))

#Converting monthly rent to categorical values

fullf$v2a1 <- as.numeric(fullf$v2a1)
fullf$v2a1 <- discretize(fullf$v2a1,
                             method = "fixed",
                             breaks = c(-Inf,0,1000,
                                        5000,10000,45000,
                                        100000,200000,
                                        300000,400000,500000,Inf),
                             labels = c("0","1","2",
                                        "3","4","5","6","7",
                                        "8","9","10"))


set.seed(123) 

# pca visualisation
pca = prcomp(fulln, center = TRUE, scale. = TRUE)
full = full %>% 
  cbind(pca$x[,1:3])
fviz_eig(pca)

fviz_pca_var(pca,
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("red", "gray", "blue"),
             select.var = list(contrib = 30), # top 30 contributing
             repel = TRUE     # Avoid text overlapping
)

#Creating partition for dataset with all factors  
trainIndex = createDataPartition(full$Target,
                                 p=0.7, list=FALSE,times=1)



trainf = fullf[trainIndex,]
testf = fullf[-trainIndex,]

#Converting meanedu into integer then decretizing it.


trainnum = fulln[trainIndex,]
testnum = fulln[-trainIndex,]

train = full[trainIndex,]
test = full[-trainIndex,]

#Baseline
base <- sum(test$Target==0)/nrow(test) #62% is the baseline
base
#Naive bayes

model <- naiveBayes(Target~.,data = trainf,
                    na.action = na.pass,laplace=1)
pred.nb<- predict(model, newdata = testf)
probs <- predict(model, testf, type="raw")


predict <- prediction(probs[,"1"], testf$Target)
perf_lr <- performance(predict, measure='tpr', x.measure='fpr')
plot(perf_lr)

error.rate.nb <- sum(testf$Target != pred.nb)/nrow(testf)
error.rate.nb
confusionMatrix(table(pred.nb,testf$Target))

#We get accuracy of 76% using all the features now we will do some 
#feature selection by  correlation and random forest and getting significance
#of each variable

#perform correlation to remove highly correlated values
costa.cor<- rcorr(as.matrix(trainnum))

#Flattens the correlation matrix to make it more interpretable
flattenCorrMatrix <- function(cormat, pmat) {
  ut <- upper.tri(cormat)
  data.frame(
    row = rownames(cormat)[row(cormat)[ut]],
    column = rownames(cormat)[col(cormat)[ut]],
    cor  =(cormat)[ut],
    p = pmat[ut]
  )
}
options(scipen=999)
costacorr <- flattenCorrMatrix(costa.cor$r, costa.cor$P)
costacorr

write.csv(costacorr,file="costarica-correlations.csv")


#based on the correlation matrix we remove correlated values
#select top & bottom 20 correlated values with Target and perform
#Naive bayes

top <- c("hogar_nin","r4t1",
         "overcrowding","r4m1","r4h1","eviv1",
         "pisocemento","epared1","hacdor","etecho1",
         "eviv2","epared2","r4m3","paredmad","energcocinar4",
         "instlevel2","tamviv","instlevel1","tamhog",
         "hogar_adul","bedrooms","lugar1","computer",
         "qmobilephone","instlevel8","rooms","v18q",
         "etecho3","paredblolad","v2a1",
         "pisomoscer","epared3","eviv3",
         "escolari","cielorazo","meaneduc","Target"
)

toprevcod <- c("tipovivi_int","lugar_int","area_int",
               "estadocivil_int","parentesco_int","instlevel_int",  
               "epared_int","etecho_int","eviv_int",
               "sanitario_int","energcocinar_int","elimbasu_int",
               "piso_int","techo_int","abasta_int","pared_int",
               "meaneduc","hogar_nin","overcrowding","rooms","escolari",
               "cielorazo","hogar_adul","Target",
               "qmobilephone","tamviv","pisomoscer","pisocemento",
               "paredblolad","v2a1","computer","hacdor")

rfvariable <- c("meaneduc","rooms","hogar_nin",
                "pisomoscer","qmobilephone","tipovivi_int",
                "lugar_int","Target")


#Visualize correlation of them
corr.top <- cor(trainnum,use = "complete.obs")
png(height=1200, width=1500, pointsize=15, file="overlap.png")
corrplot(corr.top, method="ellipse")
dev.off()
corrplot(corr.top, method="ellipse")

#Use naive bayes model with top correlated features
model2 <- naiveBayes(Target~.,data = trainf[,top], 
                    laplace=1,na.action = na.pass)

pred.nb2<- predict(model2, newdata = testf[,top])
error.rate.nb2 <- sum(test$Target != pred.nb2)/nrow(test)
error.rate.nb2
confusionMatrix(table(pred.nb2,test$Target))

#We get less accuracy of 74% since this model uses less features
#We will move forward with this

predict <- prediction(probs[,"1"], test$Target)
perf_lr <- performance(predict, measure='tpr', x.measure='fpr')
plot(perf_lr)


#Apply Random forest
rf = randomForest(Target ~., ntree = 1000,
                  data = trainf[,top],
                  na.action = na.pass)
plot(rf)
print(rf)

varImpPlot(rf,sort = T, n.var=10,main="top 10 - Variable Importance")

rf.pr = predict(rf,type="prob",newdata=testf[,top])[,2]
rfconfusionpr= predict(rf,newdata=testf[,top])

confusionMatrix(table(rfconfusionpr,testf$Target))

rf.pred = prediction(rf.pr, testf$Target)
rf.perf = performance(rf.pred,"tpr","fpr")
plot(rf.perf,main="ROC Curve for Random Forest",col=2,lwd=2)




#Logistic Regression basic
model.logit <- glm(Target ~.,family=binomial(link='logit'),
                   data=train[,top],
             na.action = na.pass )

model.logit

fitted.results <- predict(model.logit,
                          newdata=test[,top],type='response')

fitted.results <- ifelse(fitted.results > 0.5,1,0)

misClasificError <- mean(fitted.results != testnum$Target)
print(paste('Accuracy',1-misClasificError))

#77% Accuracy
probs.logit <- predict(model.logit, test[,top], type="response")



predict.logit <- prediction(fitted.results, test[,top]$Target)
perf_lr_logit <- performance(predict.logit, measure='tpr', x.measure='fpr')
plot(perf_lr_logit)

summary(model.logit)
confusionMatrix(table(fitted.results,test$Target))


