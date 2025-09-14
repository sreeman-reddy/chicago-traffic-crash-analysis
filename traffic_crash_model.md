Traffic Crash Data Model
================

``` r
library(tidyverse)
```

    ## ── Attaching core tidyverse packages ──────────────────────── tidyverse 2.0.0 ──
    ## ✔ dplyr     1.1.4     ✔ readr     2.1.5
    ## ✔ forcats   1.0.0     ✔ stringr   1.5.1
    ## ✔ ggplot2   3.5.1     ✔ tibble    3.2.1
    ## ✔ lubridate 1.9.4     ✔ tidyr     1.3.1
    ## ✔ purrr     1.0.2     
    ## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
    ## ✖ dplyr::filter() masks stats::filter()
    ## ✖ dplyr::lag()    masks stats::lag()
    ## ℹ Use the conflicted package (<http://conflicted.r-lib.org/>) to force all conflicts to become errors

``` r
library(gamlr)
```

    ## Loading required package: Matrix
    ## 
    ## Attaching package: 'Matrix'
    ## 
    ## The following objects are masked from 'package:tidyr':
    ## 
    ##     expand, pack, unpack

``` r
library(parallel)
library(distrom)
```

    ## 
    ## Attaching package: 'distrom'
    ## 
    ## The following object is masked from 'package:dplyr':
    ## 
    ##     collapse

``` r
library(caret)
```

    ## Loading required package: lattice
    ## 
    ## Attaching package: 'caret'
    ## 
    ## The following object is masked from 'package:purrr':
    ## 
    ##     lift

``` r
library(pROC)
```

    ## Type 'citation("pROC")' for a citation.
    ## 
    ## Attaching package: 'pROC'
    ## 
    ## The following objects are masked from 'package:stats':
    ## 
    ##     cov, smooth, var

``` r
library(xgboost)
```

    ## 
    ## Attaching package: 'xgboost'
    ## 
    ## The following object is masked from 'package:dplyr':
    ## 
    ##     slice

``` r
set.seed(480)
```

``` r
xnaref <- function(x){
    if(is.factor(x))
        if(!is.na(levels(x)[1]))
            x <- factor(x,levels=c(NA,levels(x)),exclude=NULL)
    return(x) }

naref <- function(DF){
    if(is.null(dim(DF))) return(xnaref(DF))
    if(!is.data.frame(DF)) 
        stop("You need to give me a data.frame or a factor")
    DF <- lapply(DF, xnaref)
    return(as.data.frame(DF))
}

roc <- function(p,y, ...){
  y <- factor(y)
  n <- length(p)
  p <- as.vector(p)
  Q <- p > matrix(rep(seq(0,1,length=100),n),ncol=100,byrow=TRUE)
  specificity <- colMeans(!Q[y==levels(y)[1],])
  sensitivity <- colMeans(Q[y==levels(y)[2],])
  plot(1-specificity, sensitivity, type="l", ...)
  abline(a=0,b=1,lty=2,col=8)
}
```

``` r
df <- read_csv("Traffic_Crashes_-_Crashes_20241108.csv")
```

    ## Rows: 890806 Columns: 48
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr (31): CRASH_RECORD_ID, CRASH_DATE_EST_I, CRASH_DATE, TRAFFIC_CONTROL_DEV...
    ## dbl (17): POSTED_SPEED_LIMIT, LANE_CNT, STREET_NO, BEAT_OF_OCCURRENCE, NUM_U...
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

``` r
drops <- c("CRASH_DATE_EST_I","LANE_CNT","INTERSECTION_RELATED_I",
           "NOT_RIGHT_OF_WAY_I","HIT_AND_RUN_I","PHOTOS_TAKEN_I",
           "STATEMENTS_TAKEN_I","DOORING_I","WORK_ZONE_I",
           "WORK_ZONE_TYPE","WORKERS_PRESENT_I", "STREET_DIRECTION",
           "STREET_NAME", "STREET_NO", "REPORT_TYPE","BEAT_OF_OCCURRENCE",
           "NUM_UNITS","INJURIES_UNKNOWN","LOCATION")
df <- df[ , !(names(df) %in% drops)]
df <- df %>% drop_na(c(LATITUDE,MOST_SEVERE_INJURY))
df$CRASH_DATE <- as.Date(df$CRASH_DATE, format = "%m/%d/%Y %I:%M:%S %p")
df$DATE_POLICE_NOTIFIED <- as.Date(df$DATE_POLICE_NOTIFIED, format = "%m/%d/%Y %I:%M:%S %p")
df <- df %>%
  mutate(SEASON = case_when(
    CRASH_MONTH %in% c(12, 1, 2) ~ 'WINTER',
    CRASH_MONTH %in% c(3, 4, 5) ~ 'SPRING',
    CRASH_MONTH %in% c(6, 7, 8) ~ 'SUMMER',
    CRASH_MONTH %in% c(9, 10, 11) ~ 'FALL'
  ))
df <- df %>% filter(LATITUDE != 0)

factor_cols <- c("TRAFFIC_CONTROL_DEVICE","DEVICE_CONDITION","WEATHER_CONDITION",
                 "LIGHTING_CONDITION","FIRST_CRASH_TYPE","TRAFFICWAY_TYPE",
                 "ALIGNMENT","ROADWAY_SURFACE_COND","ROAD_DEFECT","CRASH_TYPE",
                 "DAMAGE","PRIM_CONTRIBUTORY_CAUSE","SEC_CONTRIBUTORY_CAUSE",
                 "MOST_SEVERE_INJURY","CRASH_HOUR","CRASH_DAY_OF_WEEK","CRASH_MONTH",
                 "SEASON")
df[factor_cols] <- lapply(df[factor_cols], factor)
# df <- naref(df)
pca_ll <- prcomp(df[,c("LATITUDE","LONGITUDE")], center=TRUE, scale.=TRUE, )
df$PCLL <- pca_ll$x[,1]
df$POSTED_SPEED_LIMIT <- drop(scale(df$POSTED_SPEED_LIMIT))
```

``` r
pca_ll
```

    ## Standard deviations (1, .., p=2):
    ## [1] 1.2140180 0.7253691
    ## 
    ## Rotation (n x k) = (2 x 2):
    ##                  PC1       PC2
    ## LATITUDE   0.7071068 0.7071068
    ## LONGITUDE -0.7071068 0.7071068

``` r
sev_target <- factor(df$MOST_SEVERE_INJURY)
df_severe <- df[,!names(df) %in% c("CRASH_RECORD_ID","CRASH_DATE","DATE_POLICE_NOTIFIED","CRASH_TYPE", "MOST_SEVERE_INJURY", "TRAFFIC_CONTROL_DEVICE", "DAMAGE","INJURIES_TOTAL","INJURIES_FATAL","INJURIES_INCAPACITATING","INJURIES_NON_INCAPACITATING","INJURIES_REPORTED_NOT_EVIDENT","INJURIES_NO_INDICATION", "LATITUDE", "LONGITUDE","CRASH_HOUR","CRASH_DAY_OF_WEEK","CRASH_MONTH","SEASON", "ALIGNMENT")] #, "TRAFFICWAY_TYPE","POSTED_SPEED_LIMIT"
sev_sm <- sparse.model.matrix( ~ ., data = naref(df_severe))[,-1]
sev_sm <- sev_sm[,colSums(sev_sm)>0]
```

``` r
cl_sev = makeCluster(min(detectCores())-1)
multifit_sev <- dmr(cl_sev, sev_sm, sev_target, verb=TRUE, lmr=1e-3, cv=TRUE)
```

    ## fitting 882386 observations on 5 categories, 159 covariates.
    ## converting counts matrix to column list...
    ## distributed run.
    ## socket cluster with 9 nodes on host 'localhost'

``` r
stopCluster(cl_sev)
```

``` r
n_sev <- nrow(sev_sm)
Bdmr_sev <- coef(multifit_sev)
pdmr_sev <- predict(Bdmr_sev,sev_sm,type="response")
trueclassprobs_sev <- pdmr_sev[cbind(1:n_sev, sev_target)]
plot(trueclassprobs_sev ~ sev_target, col="lavender", varwidth=TRUE,
xlab="Severity", ylab="prob( true class )")
```

![](traffic_crash_model_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

``` r
par(mfrow=c(2,3))
for(k in names(multifit_sev)) plot(multifit_sev[[k]], main=k)
```

![](traffic_crash_model_files/figure-gfm/unnamed-chunk-7-2.png)<!-- -->

``` r
sev_pred <- predict(Bdmr_sev,sev_sm, type="class")
confusionMatrix(factor(sev_pred), factor(sev_target),mode="everything")
```

    ## Warning in levels(reference) != levels(data): longer object length is not a
    ## multiple of shorter object length

    ## Warning in confusionMatrix.default(factor(sev_pred), factor(sev_target), :
    ## Levels are not in the same order for reference and data. Refactoring data to
    ## match.

    ## Confusion Matrix and Statistics
    ## 
    ##                           Reference
    ## Prediction                  FATAL INCAPACITATING INJURY NO INDICATION OF INJURY
    ##   FATAL                         0                     0                       0
    ##   INCAPACITATING INJURY         0                     0                       0
    ##   NO INDICATION OF INJURY     687                 10185                  751936
    ##   NONINCAPACITATING INJURY    280                  4670                    6219
    ##   REPORTED, NOT EVIDENT         0                     0                       0
    ##                           Reference
    ## Prediction                 NONINCAPACITATING INJURY REPORTED, NOT EVIDENT
    ##   FATAL                                           0                     0
    ##   INCAPACITATING INJURY                           0                     0
    ##   NO INDICATION OF INJURY                     52532                 33716
    ##   NONINCAPACITATING INJURY                    17323                  4838
    ##   REPORTED, NOT EVIDENT                           0                     0
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.8718          
    ##                  95% CI : (0.8711, 0.8725)
    ##     No Information Rate : 0.8592          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.247           
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: FATAL Class: INCAPACITATING INJURY
    ## Sensitivity              0.000000                      0.00000
    ## Specificity              1.000000                      1.00000
    ## Pos Pred Value                NaN                          NaN
    ## Neg Pred Value           0.998904                      0.98316
    ## Precision                      NA                           NA
    ## Recall                   0.000000                      0.00000
    ## F1                             NA                           NA
    ## Prevalence               0.001096                      0.01684
    ## Detection Rate           0.000000                      0.00000
    ## Detection Prevalence     0.000000                      0.00000
    ## Balanced Accuracy        0.500000                      0.50000
    ##                      Class: NO INDICATION OF INJURY
    ## Sensitivity                                  0.9918
    ## Specificity                                  0.2182
    ## Pos Pred Value                               0.8856
    ## Neg Pred Value                               0.8134
    ## Precision                                    0.8856
    ## Recall                                       0.9918
    ## F1                                           0.9357
    ## Prevalence                                   0.8592
    ## Detection Rate                               0.8522
    ## Detection Prevalence                         0.9622
    ## Balanced Accuracy                            0.6050
    ##                      Class: NONINCAPACITATING INJURY
    ## Sensitivity                                  0.24799
    ## Specificity                                  0.98030
    ## Pos Pred Value                               0.51974
    ## Neg Pred Value                               0.93813
    ## Precision                                    0.51974
    ## Recall                                       0.24799
    ## F1                                           0.33577
    ## Prevalence                                   0.07917
    ## Detection Rate                               0.01963
    ## Detection Prevalence                         0.03777
    ## Balanced Accuracy                            0.61414
    ##                      Class: REPORTED, NOT EVIDENT
    ## Sensitivity                               0.00000
    ## Specificity                               1.00000
    ## Pos Pred Value                                NaN
    ## Neg Pred Value                            0.95631
    ## Precision                                      NA
    ## Recall                                    0.00000
    ## F1                                             NA
    ## Prevalence                                0.04369
    ## Detection Rate                            0.00000
    ## Detection Prevalence                      0.00000
    ## Balanced Accuracy                         0.50000

``` r
#roc curve
multiclass.roc(response=sev_target, predictor=pdmr_sev, plot=TRUE)
```

![](traffic_crash_model_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->![](traffic_crash_model_files/figure-gfm/unnamed-chunk-9-2.png)<!-- -->![](traffic_crash_model_files/figure-gfm/unnamed-chunk-9-3.png)<!-- -->![](traffic_crash_model_files/figure-gfm/unnamed-chunk-9-4.png)<!-- -->![](traffic_crash_model_files/figure-gfm/unnamed-chunk-9-5.png)<!-- -->![](traffic_crash_model_files/figure-gfm/unnamed-chunk-9-6.png)<!-- -->![](traffic_crash_model_files/figure-gfm/unnamed-chunk-9-7.png)<!-- -->![](traffic_crash_model_files/figure-gfm/unnamed-chunk-9-8.png)<!-- -->![](traffic_crash_model_files/figure-gfm/unnamed-chunk-9-9.png)<!-- -->![](traffic_crash_model_files/figure-gfm/unnamed-chunk-9-10.png)<!-- -->![](traffic_crash_model_files/figure-gfm/unnamed-chunk-9-11.png)<!-- -->![](traffic_crash_model_files/figure-gfm/unnamed-chunk-9-12.png)<!-- -->![](traffic_crash_model_files/figure-gfm/unnamed-chunk-9-13.png)<!-- -->![](traffic_crash_model_files/figure-gfm/unnamed-chunk-9-14.png)<!-- -->![](traffic_crash_model_files/figure-gfm/unnamed-chunk-9-15.png)<!-- -->![](traffic_crash_model_files/figure-gfm/unnamed-chunk-9-16.png)<!-- -->![](traffic_crash_model_files/figure-gfm/unnamed-chunk-9-17.png)<!-- -->![](traffic_crash_model_files/figure-gfm/unnamed-chunk-9-18.png)<!-- -->![](traffic_crash_model_files/figure-gfm/unnamed-chunk-9-19.png)<!-- -->![](traffic_crash_model_files/figure-gfm/unnamed-chunk-9-20.png)<!-- -->

    ## 
    ## Call:
    ## multiclass.roc.default(response = sev_target, predictor = pdmr_sev,     plot = TRUE)
    ## 
    ## Data: multivariate predictor pdmr_sev with 5 levels of sev_target: FATAL, INCAPACITATING INJURY, NO INDICATION OF INJURY, NONINCAPACITATING INJURY, REPORTED, NOT EVIDENT.
    ## Multi-class area under the curve: 0.6663

``` r
#names of top 5 variables with highest absolute coefficients
coef_sev <- coef(multifit_sev)
#coef_sev <- coef_sev[order(abs(coef_sev), decreasing = TRUE)]
exp(coef_sev[order(coef_sev[,1], decreasing = TRUE)[1:3],1])
```

    ##                               FIRST_CRASH_TYPETRAIN 
    ##                                           20.534613 
    ##                          FIRST_CRASH_TYPEPEDESTRIAN 
    ##                                           17.092708 
    ## PRIM_CONTRIBUTORY_CAUSEPHYSICAL CONDITION OF DRIVER 
    ##                                            9.793722

``` r
exp(coef_sev[order(coef_sev[,2], decreasing = TRUE)[1:3],2])
```

    ##                          FIRST_CRASH_TYPEPEDESTRIAN 
    ##                                           11.042637 
    ##                        FIRST_CRASH_TYPEPEDALCYCLIST 
    ##                                            5.886352 
    ## PRIM_CONTRIBUTORY_CAUSEPHYSICAL CONDITION OF DRIVER 
    ##                                            5.356263

``` r
exp(coef_sev[order(coef_sev[,3], decreasing = TRUE)[1:3],3])
```

    ## FIRST_CRASH_TYPESIDESWIPE SAME DIRECTION 
    ##                                 1.078354 
    ##     FIRST_CRASH_TYPEPARKED MOTOR VEHICLE 
    ##                                 1.074824 
    ##  PRIM_CONTRIBUTORY_CAUSEIMPROPER BACKING 
    ##                                 1.045759

``` r
exp(coef_sev[order(coef_sev[,4], decreasing = TRUE)[1:3],4])
```

    ##                          FIRST_CRASH_TYPEPEDESTRIAN 
    ##                                            7.136164 
    ##                        FIRST_CRASH_TYPEPEDALCYCLIST 
    ##                                            6.573600 
    ## PRIM_CONTRIBUTORY_CAUSEPHYSICAL CONDITION OF DRIVER 
    ##                                            3.226531

``` r
exp(coef_sev[order(coef_sev[,5], decreasing = TRUE)[1:3],5])
```

    ##                          FIRST_CRASH_TYPEPEDESTRIAN 
    ##                                            3.215329 
    ##                        FIRST_CRASH_TYPEPEDALCYCLIST 
    ##                                            2.388888 
    ## PRIM_CONTRIBUTORY_CAUSEPHYSICAL CONDITION OF DRIVER 
    ##                                            1.693033

``` r
dmg_target <- factor(df$DAMAGE)
df_damage <- df[,!names(df) %in% c("CRASH_RECORD_ID","CRASH_DATE","DATE_POLICE_NOTIFIED","CRASH_TYPE", "MOST_SEVERE_INJURY", "TRAFFIC_CONTROL_DEVICE", "DAMAGE","INJURIES_TOTAL","INJURIES_FATAL","INJURIES_INCAPACITATING","INJURIES_NON_INCAPACITATING","INJURIES_REPORTED_NOT_EVIDENT","INJURIES_NO_INDICATION","LATITUDE","LONGITUDE","CRASH_HOUR","CRASH_DAY_OF_WEEK","CRASH_MONTH","SEASON", "ALIGNMENT")]
dmg_sm <- sparse.model.matrix( ~ ., data = naref(df_damage))[,-1]
dmg_sm <- dmg_sm[,colSums(dmg_sm)>0]
```

``` r
names(df_damage)
```

    ##  [1] "POSTED_SPEED_LIMIT"      "DEVICE_CONDITION"       
    ##  [3] "WEATHER_CONDITION"       "LIGHTING_CONDITION"     
    ##  [5] "FIRST_CRASH_TYPE"        "TRAFFICWAY_TYPE"        
    ##  [7] "ROADWAY_SURFACE_COND"    "ROAD_DEFECT"            
    ##  [9] "PRIM_CONTRIBUTORY_CAUSE" "SEC_CONTRIBUTORY_CAUSE" 
    ## [11] "PCLL"

``` r
cl_dmg = makeCluster(min(detectCores())-1)
multifit_dmg <- dmr(cl_dmg, dmg_sm, dmg_target, verb=TRUE, lmr=1e-3,cv=TRUE)
```

    ## fitting 882386 observations on 3 categories, 159 covariates.
    ## converting counts matrix to column list...
    ## distributed run.
    ## socket cluster with 9 nodes on host 'localhost'

``` r
stopCluster(cl_dmg)
```

``` r
n_dmg <- nrow(dmg_sm)
Bdmr_dmg <- coef(multifit_dmg)
pdmr_dmg <- predict(Bdmr_dmg,dmg_sm,type="response")
trueclassprobs_dmg <- pdmr_dmg[cbind(1:n_dmg, dmg_target)]
plot(trueclassprobs_dmg ~ dmg_target, col="lavender", varwidth=TRUE,
xlab="Severity", ylab="prob( true class )")
```

![](traffic_crash_model_files/figure-gfm/unnamed-chunk-14-1.png)<!-- -->

``` r
par(mfrow=c(1,3))
for(k in names(multifit_dmg)) plot(multifit_dmg[[k]], main=k)
```

![](traffic_crash_model_files/figure-gfm/unnamed-chunk-14-2.png)<!-- -->

``` r
dmg_pred <- predict(Bdmr_dmg,dmg_sm, type="class")
confusionMatrix(factor(dmg_pred), factor(dmg_target),mode="everything")
```

    ## Confusion Matrix and Statistics
    ## 
    ##                Reference
    ## Prediction      $500 OR LESS $501 - $1,500 OVER $1,500
    ##   $500 OR LESS         18020          5089       10564
    ##   $501 - $1,500          386          1065        1074
    ##   OVER $1,500          80772        222604      542812
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.6368          
    ##                  95% CI : (0.6358, 0.6378)
    ##     No Information Rate : 0.6284          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.0744          
    ##                                           
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: $500 OR LESS Class: $501 - $1,500
    ## Sensitivity                      0.18169             0.004656
    ## Specificity                      0.98001             0.997766
    ## Pos Pred Value                   0.53515             0.421782
    ## Neg Pred Value                   0.90438             0.741217
    ## Precision                        0.53515             0.421782
    ## Recall                           0.18169             0.004656
    ## F1                               0.27128             0.009209
    ## Prevalence                       0.11240             0.259249
    ## Detection Rate                   0.02042             0.001207
    ## Detection Prevalence             0.03816             0.002862
    ## Balanced Accuracy                0.58085             0.501211
    ##                      Class: OVER $1,500
    ## Sensitivity                     0.97901
    ## Specificity                     0.07489
    ## Pos Pred Value                  0.64148
    ## Neg Pred Value                  0.67849
    ## Precision                       0.64148
    ## Recall                          0.97901
    ## F1                              0.77509
    ## Prevalence                      0.62835
    ## Detection Rate                  0.61516
    ## Detection Prevalence            0.95898
    ## Balanced Accuracy               0.52695

``` r
#roc curve
multiclass.roc(response=dmg_target, predictor=pdmr_dmg, plot=TRUE)
```

![](traffic_crash_model_files/figure-gfm/unnamed-chunk-16-1.png)<!-- -->![](traffic_crash_model_files/figure-gfm/unnamed-chunk-16-2.png)<!-- -->![](traffic_crash_model_files/figure-gfm/unnamed-chunk-16-3.png)<!-- -->![](traffic_crash_model_files/figure-gfm/unnamed-chunk-16-4.png)<!-- -->![](traffic_crash_model_files/figure-gfm/unnamed-chunk-16-5.png)<!-- -->![](traffic_crash_model_files/figure-gfm/unnamed-chunk-16-6.png)<!-- -->

    ## 
    ## Call:
    ## multiclass.roc.default(response = dmg_target, predictor = pdmr_dmg,     plot = TRUE)
    ## 
    ## Data: multivariate predictor pdmr_dmg with 3 levels of dmg_target: $500 OR LESS, $501 - $1,500, OVER $1,500.
    ## Multi-class area under the curve: 0.6424

``` r
coef_dmg <- coef(multifit_dmg)
exp(coef_dmg[order(coef_dmg[,1], decreasing = TRUE)[1:3],1])
```

    ##   FIRST_CRASH_TYPEPEDESTRIAN FIRST_CRASH_TYPEPEDALCYCLIST 
    ##                     5.500340                     4.724711 
    ##       FIRST_CRASH_TYPEANIMAL 
    ##                     2.212956

``` r
exp(coef_dmg[order(coef_dmg[,2], decreasing = TRUE)[1:3],2])
```

    ##              TRAFFICWAY_TYPEPARKING LOT   PRIM_CONTRIBUTORY_CAUSENOT APPLICABLE 
    ##                                1.271567                                1.220511 
    ## PRIM_CONTRIBUTORY_CAUSEIMPROPER BACKING 
    ##                                1.210166

``` r
exp(coef_dmg[order(coef_dmg[,3], decreasing = TRUE)[1:3],3])
```

    ##                                                                FIRST_CRASH_TYPEOVERTURNED 
    ##                                                                                  1.226727 
    ##                                                              FIRST_CRASH_TYPEFIXED OBJECT 
    ##                                                                                  1.176215 
    ## PRIM_CONTRIBUTORY_CAUSEUNDER THE INFLUENCE OF ALCOHOL/DRUGS (USE WHEN ARREST IS EFFECTED) 
    ##                                                                                  1.168072

``` r
sev_target_encoded <- ifelse(sev_target=="FATAL",0,ifelse(sev_target=="INCAPACITATING INJURY",1,ifelse(sev_target=="NO INDICATION OF INJURY",2,ifelse(sev_target=="NONINCAPACITATING INJURY",3,4))))
sev_param <- list("objective" = "multi:softmax", "num_class" = 5, "eval_metric"="mlogloss")
XGB_sev <- xgboost(data = sev_sm,
                  label = sev_target_encoded,
                  param=sev_param,
                  nrounds=50,
                  verbose = FALSE,
                  nthread = 10)
```

``` r
sev_pred_xgb_encoded <- predict(XGB_sev,sev_sm)
sev_pred_xgb <- ifelse(sev_pred_xgb_encoded==0,"FATAL",ifelse(sev_pred_xgb_encoded==1,"INCAPACITATING INJURY",ifelse(sev_pred_xgb_encoded==2,"NO INDICATION OF INJURY",ifelse(sev_pred_xgb_encoded==3,"NONINCAPACITATING INJURY","REPORTED, NOT EVIDENT"))))
confusionMatrix(factor(sev_pred_xgb), factor(sev_target),mode="everything")
```

    ## Confusion Matrix and Statistics
    ## 
    ##                           Reference
    ## Prediction                  FATAL INCAPACITATING INJURY NO INDICATION OF INJURY
    ##   FATAL                        16                     0                       0
    ##   INCAPACITATING INJURY         1                    57                       4
    ##   NO INDICATION OF INJURY     672                 10112                  752038
    ##   NONINCAPACITATING INJURY    278                  4686                    6109
    ##   REPORTED, NOT EVIDENT         0                     0                       4
    ##                           Reference
    ## Prediction                 NONINCAPACITATING INJURY REPORTED, NOT EVIDENT
    ##   FATAL                                           0                     0
    ##   INCAPACITATING INJURY                           4                     2
    ##   NO INDICATION OF INJURY                     52189                 33615
    ##   NONINCAPACITATING INJURY                    17660                  4920
    ##   REPORTED, NOT EVIDENT                           2                    17
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.8724          
    ##                  95% CI : (0.8717, 0.8731)
    ##     No Information Rate : 0.8592          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.2522          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: FATAL Class: INCAPACITATING INJURY
    ## Sensitivity             1.655e-02                    3.837e-03
    ## Specificity             1.000e+00                    1.000e+00
    ## Pos Pred Value          1.000e+00                    8.382e-01
    ## Neg Pred Value          9.989e-01                    9.832e-01
    ## Precision               1.000e+00                    8.382e-01
    ## Recall                  1.655e-02                    3.837e-03
    ## F1                      3.255e-02                    7.639e-03
    ## Prevalence              1.096e-03                    1.684e-02
    ## Detection Rate          1.813e-05                    6.460e-05
    ## Detection Prevalence    1.813e-05                    7.706e-05
    ## Balanced Accuracy       5.083e-01                    5.019e-01
    ##                      Class: NO INDICATION OF INJURY
    ## Sensitivity                                  0.9919
    ## Specificity                                  0.2225
    ## Pos Pred Value                               0.8862
    ## Neg Pred Value                               0.8188
    ## Precision                                    0.8862
    ## Recall                                       0.9919
    ## F1                                           0.9361
    ## Prevalence                                   0.8592
    ## Detection Rate                               0.8523
    ## Detection Prevalence                         0.9617
    ## Balanced Accuracy                            0.6072
    ##                      Class: NONINCAPACITATING INJURY
    ## Sensitivity                                  0.25281
    ## Specificity                                  0.98032
    ## Pos Pred Value                               0.52477
    ## Neg Pred Value                               0.93850
    ## Precision                                    0.52477
    ## Recall                                       0.25281
    ## F1                                           0.34123
    ## Prevalence                                   0.07917
    ## Detection Rate                               0.02001
    ## Detection Prevalence                         0.03814
    ## Balanced Accuracy                            0.61656
    ##                      Class: REPORTED, NOT EVIDENT
    ## Sensitivity                             4.409e-04
    ## Specificity                             1.000e+00
    ## Pos Pred Value                          7.391e-01
    ## Neg Pred Value                          9.563e-01
    ## Precision                               7.391e-01
    ## Recall                                  4.409e-04
    ## F1                                      8.814e-04
    ## Prevalence                              4.369e-02
    ## Detection Rate                          1.927e-05
    ## Detection Prevalence                    2.607e-05
    ## Balanced Accuracy                       5.002e-01

``` r
sev_importance <- xgb.importance(model = XGB_sev,feature_names = colnames(sev_sm))
t10_imp_sev <- head(sev_importance,10)
t10_imp_sev
```

    ##                                                 Feature       Gain      Cover
    ##                                                  <char>      <num>      <num>
    ##  1:                          FIRST_CRASH_TYPEPEDESTRIAN 0.32358646 0.06623153
    ##  2:                        FIRST_CRASH_TYPEPEDALCYCLIST 0.13967915 0.04450819
    ##  3:                FIRST_CRASH_TYPEPARKED MOTOR VEHICLE 0.11376201 0.05472634
    ##  4:            FIRST_CRASH_TYPESIDESWIPE SAME DIRECTION 0.06852385 0.03937016
    ##  5:                                                PCLL 0.04084961 0.06359608
    ##  6:                          TRAFFICWAY_TYPEPARKING LOT 0.02062636 0.02842781
    ##  7:             PRIM_CONTRIBUTORY_CAUSEIMPROPER BACKING 0.01821197 0.02196397
    ##  8: PRIM_CONTRIBUTORY_CAUSEPHYSICAL CONDITION OF DRIVER 0.01801134 0.03987643
    ##  9:                               FIRST_CRASH_TYPEANGLE 0.01755259 0.01428394
    ## 10:                            FIRST_CRASH_TYPEREAR END 0.01753146 0.01631573
    ##       Frequency
    ##           <num>
    ##  1: 0.017197609
    ##  2: 0.011055606
    ##  3: 0.013102940
    ##  4: 0.008598804
    ##  5: 0.283842437
    ##  6: 0.013839980
    ##  7: 0.007452297
    ##  8: 0.018098436
    ##  9: 0.006551470
    ## 10: 0.010646139

``` r
gp = xgb.ggplot.importance(t10_imp_sev)
print(gp)
```

![](traffic_crash_model_files/figure-gfm/unnamed-chunk-21-1.png)<!-- -->

``` r
dmg_target_encoded <- ifelse(dmg_target=="$500 OR LESS",0,ifelse(dmg_target=="$501 - $1,500",1,2))
XGB_dmg <- xgboost(data = dmg_sm,
                  label = dmg_target_encoded,
                  objective = "multi:softmax",
                  num_class = 3,
                  #eval="mlogloss",
                  nrounds=50,
                  #nfold=5,
                  verbose = FALSE,
                  #prediction = TRUE,
                  nthread = 10)
```

``` r
dmg_pred_xgb_encoded <- predict(XGB_dmg,dmg_sm)
dmg_pred_xgb <- ifelse(dmg_pred_xgb_encoded==0,"$500 OR LESS",ifelse(dmg_pred_xgb_encoded==1,"$501 - $1,500","OVER $1,500"))
confusionMatrix(factor(dmg_pred_xgb), factor(dmg_target),mode="everything")
```

    ## Confusion Matrix and Statistics
    ## 
    ##                Reference
    ## Prediction      $500 OR LESS $501 - $1,500 OVER $1,500
    ##   $500 OR LESS         18011          4937        9843
    ##   $501 - $1,500         1492          8977        4166
    ##   OVER $1,500          79675        214844      540441
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.6431          
    ##                  95% CI : (0.6421, 0.6441)
    ##     No Information Rate : 0.6284          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.1008          
    ##                                           
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: $500 OR LESS Class: $501 - $1,500
    ## Sensitivity                      0.18160              0.03924
    ## Specificity                      0.98113              0.99134
    ## Pos Pred Value                   0.54927              0.61339
    ## Neg Pred Value                   0.90446              0.74672
    ## Precision                        0.54927              0.61339
    ## Recall                           0.18160              0.03924
    ## F1                               0.27296              0.07377
    ## Prevalence                       0.11240              0.25925
    ## Detection Rate                   0.02041              0.01017
    ## Detection Prevalence             0.03716              0.01659
    ## Balanced Accuracy                0.58137              0.51529
    ##                      Class: OVER $1,500
    ## Sensitivity                      0.9747
    ## Specificity                      0.1019
    ## Pos Pred Value                   0.6473
    ## Neg Pred Value                   0.7046
    ## Precision                        0.6473
    ## Recall                           0.9747
    ## F1                               0.7779
    ## Prevalence                       0.6284
    ## Detection Rate                   0.6125
    ## Detection Prevalence             0.9463
    ## Balanced Accuracy                0.5383

``` r
dmg_importance <- xgb.importance(model = XGB_dmg,feature_names = colnames(dmg_sm))
t10_imp_dmg <- head(dmg_importance,10)
t10_imp_dmg
```

    ##                                      Feature       Gain      Cover   Frequency
    ##                                       <char>      <num>      <num>       <num>
    ##  1:               FIRST_CRASH_TYPEPEDESTRIAN 0.20193039 0.03674903 0.010362063
    ##  2:             FIRST_CRASH_TYPEPEDALCYCLIST 0.10857484 0.02023357 0.006704864
    ##  3:                    FIRST_CRASH_TYPEANGLE 0.07634860 0.03095280 0.010118249
    ##  4:                                     PCLL 0.07328627 0.09087479 0.267097403
    ##  5:                  FIRST_CRASH_TYPETURNING 0.04962121 0.02445003 0.009021090
    ##  6:             FIRST_CRASH_TYPEFIXED OBJECT 0.03874662 0.02677145 0.011337316
    ##  7: LIGHTING_CONDITIONDARKNESS, LIGHTED ROAD 0.03490729 0.01619953 0.020846032
    ##  8:               TRAFFICWAY_TYPEPARKING LOT 0.03059386 0.02371356 0.015604047
    ##  9:                  TRAFFICWAY_TYPEFOUR WAY 0.02912239 0.01953234 0.008899183
    ## 10:  PRIM_CONTRIBUTORY_CAUSEIMPROPER BACKING 0.01986497 0.02401918 0.011581129

``` r
gp = xgb.ggplot.importance(t10_imp_dmg)
print(gp)
```

![](traffic_crash_model_files/figure-gfm/unnamed-chunk-25-1.png)<!-- -->
