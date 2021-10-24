rm(list=ls()) #borro todas las variables del workspace (rm)
rtas <- read.csv("respuestas.csv",sep = "\t",header = T, quote = "",colClasses = c("integer", "integer", "integer", "character", "character","character"))
seqs <- read.csv("secuencias.csv",sep = "\t",header = T, colClasses = c("integer", "integer", "character"))
sujs <- read.csv("sujetos.csv",sep = "\t",header = T, colClasses = c("integer", "character", "integer","factor","character","integer","character"))
txts <- read.csv("textos.csv",sep = "\t",header = T, colClasses = c("integer","integer","integer","character"))
mailsDlt <- c('brunobian@gmail.com', 'diegoshalom@gmail.com', 'test1@test.com', 'test2@test.com', 'test4@test.com', 'test6@test.com', 'test10@test.com', 'test11@test.com', 'test12@test.com', 'test13@test.com', 'test14@test.com', 'test21@test.com', 'test22@test.com', 'juan@test.com', 'test@test.com', 'test23@test.com', 'juan@test', 'subjuntivo%40gmail.com', 'matias_pablo99@hotmail.com', 'lililibreria1@gmail.com', 'laalopy_baez@hotmail.com',  'majo218@hotmail.com',  'franciscobonafine@hotmail.com', 'yoboludeo@ymail.com', 'Uwamataru@hotmail.com', 'denistula10@gmail.com', 'guada_17@live.com', 'Darksoul444@yopmail.com', 'albanyreyes31@hotmail.com', 'belen_16_27@hotmail.com.ar', 'caroline-medina-uruguay@hotmail.com', 'dali.28218@gmail.com', 'duoresplandor1@yahoo.com', 'floabreo@hotmail.com', 'ignaciolinarim@hotmail.com', 'news_goldens@hotmail.com', 'verofarias10@yahoo.com', 'test6@test.com', 'pipi_brian@hotmail.es', 'tinchosalto7@hotmail.com', 'fdh@asd', 'pipi_brian@hotmail.es', 'segoviajorge_2089@hotmail.com', 'emilianomelero@gmail.com', 'roohflekiifernandez@outlook.com', 'dfslezak+aa@gmail.com')
rtas$words[2]

library(stringi)
library(dplyr)

# Para cada sujeto y cuento me quedo con el ultimo registro
rtas2 <- rtas %>%
  group_by_(.dots=c("subject_id"," trialOpt_id")) %>%
  dplyr::summarize(words = last(words))

# Edito las respuestas para quedarme solo con las palabras

init = T
for (i in 1:dim(rtas2)[1]){
  
  # i=4365
  # 3495
  # 3906
  
  if (i%%100 == 0){print(i)}
  
  x <- rtas2[i,]
  thisMail <- sujs$email[sujs$id == x$subject_id]
  if (!is.element(thisMail,mailsDlt)){
    x$words <- gsub("\\[","",x$words)
    x$words <- gsub("\\]","",x$words)
    x$words <- gsub('\\"',"",x$words)
    x$words <- gsub(", [0-9]+","",x$words)
    x$words <- gsub("\\\\","\\",x$words,fixed=TRUE)
    x$words <- gsub(", ,",",",x$words,fixed=TRUE)
    x$words <- tolower(x$words)
    x$words <- stri_trans_general(x$words,"latin-ascii")
    x$words <- stri_unescape_unicode(x$words)
    respuestas <- strsplit(x$words,", ")[[1]]
    
    # Busco la secuencia de palabras pedidas
    thisSeq  <- seqs[seqs$id == x$trialOpt_id,]
    misWords <- thisSeq$missing_words
    misWords <- gsub("\\[","",misWords)
    misWords <- gsub("\\]","",misWords)
    misWords <- sapply(strsplit(misWords,",")[[1]], as.integer)
    misWords <- misWords+1
    
    misWords <- misWords[1:length(respuestas)]      

    if (is.element(NA,misWords)){
      print(i)
    }
    
    # Busco las palabras pedidas
    t <- txts$body[txts$id == thisSeq$text_id]
    t <- gsub("[-.,¡!?¿();]","",t)
    t <- gsub("\\[","",t)
    t <- gsub("\\]","",t)
    t <- gsub(":","",t)
    t <- gsub("</p>","",t)
    t <- gsub("<p>","",t)
    t <- gsub("\u0085","",t)
    t <- tolower(t)
    t <- strsplit(t," ")[[1]]
    t <- stri_trans_general(t,"latin-ascii")
    originales <- t[misWords]
    originales <- tolower(originales)
    originales <- stri_trans_general(originales,"latin-ascii")
    
    tId <- rep(thisSeq$text_id, length(respuestas))
    
    tmp <- data.frame(tId, misWords, originales, respuestas, stringsAsFactors=F)
    tmp$iguales <- tmp$originales == tmp$respuestas
    
    if (init){
      df <- tmp
      init <- F
    }else{
      df <- rbind(df, tmp)
    }
  }
}

df2 <- df %>%
  group_by_(.dots=c("tId", "misWords")) %>%
  dplyr::summarize(originales = last(originales),
                   pred = mean(iguales),
                   nPred = n(),
                   words = paste(respuestas, collapse = ','))

write.csv(df2, file = "predictability.csv")


# Cuento cantidad de respuestas por sujeto
countCharOccurrences <- function(char, s) {
  s2 <- gsub(char,"",s)
  return (nchar(s) - nchar(s2))
}

argupo1 <- rtas2 %>%
  group_by_(.dots=c("subject_id"," trialOpt_id")) %>%
  dplyr::summarize(nrtas = countCharOccurrences(',',words))

argupo1 <- argupo1[argupo1$nrtas>100,]

agrupo2 <- argupo1 %>%
  group_by_(.dots=c("subject_id")) %>%
  dplyr::summarize(nrtas = sum(nrtas),
                   ntexts = n())

mean(agrupo2$nrtas)
mean(agrupo2$ntexts)
