rm(list=ls())

## Load libraries
library(dplyr)
library(rstudioapi)
library(tidyverse)
library(GGally)
library(data.table)
library(moments)

options(scipen=999)  ## disattiva l'annotazione scientifica

current_path <- getActiveDocumentContext()$path
setwd(dirname(current_path ))
print(getwd())

df <- read_csv("UCI_Credit_Card.csv")
str(df)

df <- df %>%
  rename(default = `default.payment.next.month`) %>%
  mutate(
    default = as.factor(default),
    SEX = as.factor(SEX),
    EDUCATION = as.factor(EDUCATION),
    MARRIAGE = as.factor(MARRIAGE)
  )


# Calcoliamo le percentuali per la variabile target (default)
tab <- df$default %>% table()
percentages <- tab %>% prop.table() %>% round(3) * 100
percentages
## Create text labels (0 = No Default, 1 = Default)
txt <- paste0(c("No Default (0)", "Default (1)"), '\n', percentages, "%")
pie(tab, labels=txt, main="Distribuzione della variabile Target (Default)")

### Barplot 
#barplot per il livello di istruzione
bb <- df$EDUCATION %>% table() %>%
  barplot(main="Distribution for Level of Istruction",
          ylab = "Frequency",
          col=c("pink", "lightblue", "lightgreen", "orange", "purple", "grey", "darkred"))


### Check for missing values per column
colSums(is.na(df))

# Vediamo se ci sono righe duplicate (escludendo l'ID che è sempre unico)
df_no_id <- df %>% select(-ID)
sum(duplicated(df_no_id)) # Conta i duplicati
df <- distinct(df_no_id) # Se ci sono duplicati, li rimuoviamo

# Raggruppamento categorie
df <- df %>%
  mutate(
    EDUCATION = as.character(EDUCATION),
    EDUCATION = ifelse(EDUCATION %in% c("0", "4", "5", "6"), "4", EDUCATION),
    EDUCATION = as.factor(EDUCATION),
    
    MARRIAGE = as.character(MARRIAGE),
    MARRIAGE = ifelse(MARRIAGE == "0", "3", MARRIAGE),
    MARRIAGE = as.factor(MARRIAGE)
  )

#GGplot sesso-default
ggplot(df, aes(x = SEX, fill = default)) +
  geom_bar(position = "fill") + # "fill" mostra le proporzioni percentuali
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Default rate for Sex", x = "Sex (1 = Man, 2 = Woman)", y = "Percentage") +
  theme_minimal()


# GGplot education vs Default
ggplot(df, aes(x = EDUCATION, fill = default)) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Default rate for education", x = "Level of education", y = "Percentage") +
  theme_minimal()


# GGplot marriage vs Default
ggplot(df, aes(x = MARRIAGE, fill = default)) +
  geom_bar(position = "fill") + 
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Default rate for marriage", x = "Marriage status", y = "Percentage") +
  theme_minimal()

# Distribuzione del limite di credito rispetto al default
ggplot(df, aes(x = default, y = LIMIT_BAL, fill = default)) +
  geom_boxplot() +
  labs(title = "LIMIT_BAL Distribution for Default State", x = "Default", y = "Credit Limit") +
  theme_minimal()


## ANALISI DELLA CORRELAZIONE
library(corrplot)
num_vars <- df %>% select(LIMIT_BAL, AGE, starts_with("BILL_AMT"), starts_with("PAY_AMT")) # Abbiamo selezionato solo le variabili numeriche continue

#matrice di correlazione
cor_matrix <- cor(num_vars, use = "complete.obs")

corrplot(cor_matrix, method = "color", type = "upper",
         tl.cex = 0.7, addCoef.col = "black", number.cex = 0.5,
         title = " Correlation Matrix for Continuous Variables", mar=c(0,0,1,0))


## Ggplot dello storico dei pagamenti (PAY_0 ... PAY_6)
ggplot(df, aes(x = as.factor(PAY_0), fill = default)) +
  geom_bar() +
  labs(title = "Stato di pagamento più recente (PAY_0) vs Default",
       x = "Stato Pagamento (Mesi di ritardo)", y = "Conteggio") +
  theme_minimal()

# NB: ggpairs su 30.000 righe e 25 variabili bloccherebbe il PC.
#quindi campioniamo 1000 righe e selezioniamo le variabili più importanti
df_sample <- df %>% sample_n(1000) %>% select(default, LIMIT_BAL, AGE, PAY_0, BILL_AMT1)
ggpairs(df_sample, aes(color=default, alpha=0.5))


### STATISTICHE ROBUSTE: Standard e Trimmed Mean 
df %>% pull(LIMIT_BAL) %>% mean(na.rm = TRUE) #media standard del limite di credito
df %>% pull(LIMIT_BAL) %>% mean(trim = 0.025, na.rm = TRUE) # media troncata escludendo 
#il 2.5% dei limiti di credito più bassi e il 2.5% di quelli più alti (tecnica se ci sono outlier enormi nei dati) 



###CREAZIONE DI NUOVA VARIABILE (Uso del credito)
# Aggiungiamo un piccolo valore (es. 1) al denominatore per evitare divisioni per zero
df <- df %>%
  mutate(
    UTIL_1 = BILL_AMT1 / (LIMIT_BAL+1),
    UTIL_2 = BILL_AMT2 / (LIMIT_BAL+1),
    UTIL_3 = BILL_AMT3 / (LIMIT_BAL+1),
    UTIL_4 = BILL_AMT4 / (LIMIT_BAL+1),
    UTIL_5 = BILL_AMT5 / (LIMIT_BAL+1),
    UTIL_6 = BILL_AMT6 / (LIMIT_BAL+1)
  )

#Trovo il picco
df <- df %>%
  mutate(
    MAX_UTIL = pmax(UTIL_1, UTIL_2, UTIL_3, UTIL_4, UTIL_5, UTIL_6))


# Aggiornamento del boxplot
ggplot(df, aes(x = default, y = MAX_UTIL, fill = default)) +
  geom_boxplot() +
  coord_cartesian(ylim = c(0, 1.5)) + # Limitiamo l'asse y per gestire outlier estremi
  labs(title = "MAX_UTIL vs Default",
       x = "Default (0 = No, 1 = Yes)",
       y = "Maximum usage within 6 months") +
  theme_minimal()


## DIVISIONE TRA TRAIN E TEST
library(caret)
set.seed(42)

train_index <- createDataPartition(df$default, p = 0.8, list = FALSE) # createDataPartition bilancia automaticamente la proporzione di 0 e 1 nei due set

df_train <- df[train_index, ]
df_test  <- df[-train_index, ]

#Stampa delle dimensioni
cat("Dimensioni Training Set:", dim(df_train), "\n")
cat("Dimensioni Test Set:", dim(df_test), "\n")

prop.table(table(df_train$default)) #verifica della coerenza
prop.table(table(df_test$default))

###STANDARDIZZAZIONE DELLE VARIABILI CONTINUE 
vars_to_scale <- c("LIMIT_BAL", "AGE",
                   "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
                   "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
                   "UTIL_1", "UTIL_2", "UTIL_3", "UTIL_4", "UTIL_5", "UTIL_6", "MAX_UTIL")

# calcolo della media e della deviazione standard (train set)
preProcValues <- preProcess(df_train[, vars_to_scale], method = c("center", "scale"))

df_train_scaled <- df_train
df_test_scaled <- df_test

df_train_scaled[, vars_to_scale] <- predict(preProcValues, df_train[, vars_to_scale])

df_test_scaled[, vars_to_scale]  <- predict(preProcValues, df_test[, vars_to_scale])

cat("\nVerifica Standardizzazione su LIMIT_BAL (Train):\nMedia:",
    round(mean(df_train_scaled$LIMIT_BAL), 5), "\nDev.Std:", sd(df_train_scaled$LIMIT_BAL), "\n")

### PCA 
bill_train <- df_train_scaled %>% select(starts_with("BILL_AMT"))

# Calcolo della PCA (train set)
pca_bills <- prcomp(bill_train, center = FALSE, scale. = FALSE)
summary(pca_bills)

df_train_scaled$BILL_PC1 <- pca_bills$x[, 1] #aggiunta del pc1 al train set

#PCA al test set 
bill_test <- df_test_scaled %>% select(starts_with("BILL_AMT"))
df_test_scaled$BILL_PC1 <- predict(pca_bills, newdata = bill_test)[, 1]

#Rimuoviamo le variabili ridondanti ( BILL e UTIL)
vars_to_remove <- c("BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
                    "UTIL_1", "UTIL_2", "UTIL_3", "UTIL_4", "UTIL_5", "UTIL_6")

df_train_final <- df_train_scaled %>% select(-all_of(vars_to_remove))
df_test_final  <- df_test_scaled %>% select(-all_of(vars_to_remove))
str(df_train_final)


### REGRESSIONE LOGISTICA UNBALANCED
logistic_unbalanced <- glm(default ~ ., data = df_train_final, family = "binomial")
summary(logistic_unbalanced)

#Calcolo odds ratio
odds_ratios_u <- exp(coef(logistic_unbalanced))
print(odds_ratios_u)

#Previsione sul test set
prob_test_u <- predict(logistic_unbalanced, newdata = df_test_final, type = "response")

#Conversione delle probabilità in classi
soglia_u <- 0.22
pred_class_u <- ifelse(prob_test_u >= soglia_u, 1, 0)

#CONFUSION MATRIX 
actual_class_u <- as.numeric(as.character(df_test_final$default))

confusion_matrix_u <- table(pred_class_u, actual_class_u)
print(confusion_matrix_u)

# MISCLASSIFICATION ERROR RATE 
error_rate_u <- 1 - (sum(diag(confusion_matrix_u)) / sum(confusion_matrix_u))
cat("\nMisclassification Error Rate (Test Set):", round(error_rate_u, 4), "\n")

sensitivity_u <- confusion_matrix_u[2, 2] / sum(confusion_matrix_u[, 2])
print(sensitivity_u)

accuracy_u <- sum(diag(confusion_matrix_u)) / sum(confusion_matrix_u)
print(accuracy_u)

specificity_u <- confusion_matrix_u[1, 1] / sum(confusion_matrix_u[, 1])
print(specificity_u)


library(pROC)
roc_curve_u <- roc(df_test_final$default, prob_test_u)
#AUC
auc_value_u <- auc(roc_curve_u)
print(auc_value_u)

plot(roc_curve_u,
     main = paste("Curva ROC (AUC =", round(auc_value_u, 4), ")"),
     col = "#1c86ee",
     lwd = 3,
     identity = TRUE,
     identity.col = "gray",
     identity.lty = 2,
     print.thres = "best") 


# Ribilanciamento del training set (ROSE)
library(ROSE)

set.seed(42)

df_train_balanced <- ROSE(default ~ ., data = df_train_final, seed = 42)$data #default ~ . significa "bilancia rispetto a tutte le colonne rimaste")

#Verifica del bilanciamento
print(table(df_train_final$default))

print(table(df_train_balanced$default))


### REGRESSIONE LOGISTICA BILANCIATA
m_logistica <- glm(default ~ ., data = df_train_balanced, family = "binomial")
library(car)
vif(m_logistica)
summary(m_logistica)

#Calcolo odds ratio
odds_ratios <- exp(coef(m_logistica))
print(odds_ratios)

#Previsione sul test set
prob_test <- predict(m_logistica, newdata = df_test_final, type = "response")

#Conversione delle probabilità in classi
soglia <- 0.5
pred_class <- ifelse(prob_test >= soglia, 1, 0)

#CONFUSION MATRIX 
actual_class <- as.numeric(as.character(df_test_final$default))

confusion_matrix <- table(pred_class, actual_class)
print(confusion_matrix)

# MISCLASSIFICATION ERROR RATE 
error_rate <- 1 - (sum(diag(confusion_matrix)) / sum(confusion_matrix))
cat("\nMisclassification Error Rate (Test Set):", round(error_rate, 4), "\n")

sensitivity <- confusion_matrix[2, 2] / sum(confusion_matrix[, 2])
print(sensitivity)

accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(accuracy)

specificity <- confusion_matrix[1, 1] / sum(confusion_matrix[, 1])
print(specificity)


library(pROC)
roc_curve <- roc(df_test_final$default, prob_test)
#AUC
auc_value <- auc(roc_curve)
print(auc_value)

plot(roc_curve,
     main = paste("Curva ROC (AUC =", round(auc_value, 4), ")"),
     col = "#1c86ee",
     lwd = 3,
     identity = TRUE,
     identity.col = "gray",
     identity.lty = 2,
     print.thres = "best") 


### STEPWISE REGRESSION 
m_stepwise <- step(logistic_unbalanced, direction = "both", trace = FALSE)
summary(m_stepwise)

### Nuove metriche (stepwise)
#Test Set
prob_step <- predict(m_stepwise, newdata = df_test_final, type = "response")
pred_class_step <- ifelse(prob_step >= 0.22, 1, 0)

#Nuova matrice di confusione
confusion_matrix_step <- table(pred_class_step, actual_class)
print(confusion_matrix_step)

accuracy_step <- sum(diag(confusion_matrix_step)) / sum(confusion_matrix_step)
print(accuracy_step)
error_rate_step <- 1 - accuracy_step
print(error_rate_step)
sensitivity_step <- confusion_matrix_step[2, 2] / sum(confusion_matrix_step[, 2])
print(sensitivity_step)
specificity_step <- confusion_matrix_step[1, 1] / sum(confusion_matrix_step[, 1])
print(specificity_step)

roc_step <- roc(df_test_final$default, prob_step)
auc_step <- auc(roc_step)
print(auc_step)

plot(roc_step,
     main = paste("Curva ROC - Stepwise Model (AUC =", round(auc_step, 4), ")"),
     col = "#2ecc71", # Colore verde per distinguerla dalla precedente
     lwd = 3,
     identity = TRUE,
     identity.col = "gray",
     identity.lty = 2,
     print.thres = "best")


### AVERAGE PARTIAL EFFECTS (APE)
library(margins)

effetti_marginali <- margins(m_stepwise)
ape_summary <- summary(effetti_marginali)
print(ape_summary)

# Ordiniamo per impatto assoluto (dal coefficiente che sposta più la probabilità a quello meno influente)
print(ape_summary %>% select(factor, AME) %>% arrange(desc(abs(AME))))


###RANDOM FOREST 
library(randomForest)

set.seed(42)
m_rf <- randomForest(default ~ .,
                     data = df_train_final,
                     ntree = 300,
                     importance = TRUE) 

print(m_rf)

#errore del modello in base al numero di alberi creati 
dev.new()
plot(m_rf, main="Errore della Random Forest vs Numero di Alberi")

dev.new()
varImpPlot(m_rf,
           main = "Importanza delle Variabili (Random Forest)",
           col = "darkred",
           pch = 16)


### Previsioni(RANDOM FOREST) 

#Test set
pred_class_rf <- predict(m_rf, newdata = df_test_final, type = "response")
prob_test_rf <- predict(m_rf, newdata = df_test_final, type = "prob")[, "1"]


# MATRICE DI CONFUSIONE
confusion_matrix_rf <- table(pred_class_rf, actual_class)
print(confusion_matrix_rf)

accuracy_rf <- sum(diag(confusion_matrix_rf)) / sum(confusion_matrix_rf)
error_rate_rf <- 1 - accuracy_rf
sensitivity_rf <- confusion_matrix_rf[2, 2] / sum(confusion_matrix_rf[, 2])
specificity_rf <- confusion_matrix_rf[1, 1] / sum(confusion_matrix_rf[, 1])


cat("\nAccuracy:", round(accuracy_rf, 4))
cat("\nError Rate:", round(error_rate_rf, 4))
cat("\nSensitivity (Tasso Veri Default intercettati):", round(sensitivity_rf, 4))
cat("\nSpecificity (Tasso Veri Sani intercettati):", round(specificity_rf, 4), "\n")

#ROC E AUC (RANDOM FOREST)
roc_rf <- roc(df_test_final$default, prob_test_rf)
auc_rf <- auc(roc_rf)
cat("\nAUC Random Forest:", round(auc_rf, 4), "\n")


#OTTIMIZZAZIONE DELLA SOGLIA PER LA RANDOM FOREST
rf_best_threshold <- coords(roc_rf, "best", ret = c("threshold", "accuracy", "sensitivity", "specificity"), best.method="youden")

cat("\nLa soglia ottimale per la Random Forest non è 0.5, ma:", round(rf_best_threshold$threshold, 4), "\n")

#Ricalcolo delle classi predette (nuova soglia)
pred_class_rf_ottimizzata <- ifelse(prob_test_rf >= 0.235, 1, 0)

confusion_matrix_rf_ottimizzata <- table(pred_class_rf_ottimizzata, actual_class)
print(confusion_matrix_rf_ottimizzata)


accuracy_rf_ottimizzata <- sum(diag(confusion_matrix_rf_ottimizzata)) / sum(confusion_matrix_rf_ottimizzata)
sensitivity_rf_ottimizzata <- confusion_matrix_rf_ottimizzata[2, 2] / sum(confusion_matrix_rf_ottimizzata[, 2])
specificity_rf_ottimizzata <- confusion_matrix_rf_ottimizzata[1, 1] / sum(confusion_matrix_rf_ottimizzata[, 1])

cat("\nNuova Accuracy:", round(accuracy_rf_ottimizzata, 4))
cat("\nNuova Sensitivity (Tasso Veri Default intercettati):", round(sensitivity_rf_ottimizzata, 4))
cat("\nNuova Specificity (Tasso Veri Sani intercettati):", round(specificity_rf_ottimizzata, 4), "\n")



### CONFRONTO  (LOGISTICA vs RF)
dev.new()

plot(roc_step, col = "orange", lwd = 2, legacy.axes = TRUE,
     main = "Confronto Curve ROC: Logistica Stepwise vs Random Forest")

plot(roc_rf, col = "darkgreen", lwd = 2, add = TRUE)

legend("bottomright",
       legend = c(paste("Logistica Stepwise (AUC =", round(auc_value, 3), ")"),
                  paste("Random Forest (AUC =", round(auc_rf, 3), ")")),
       col = c("orange", "darkgreen"), lwd = 2, bty = "n")


