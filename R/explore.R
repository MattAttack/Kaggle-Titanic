library(ggplot2)

get.data <- function() {
    df <- read.csv('../data/train.csv')
    df$Survived <- factor(df$Survived)
    df
}

plot.data <- function() {
    df <- get.data()
    p <- ggplot(df, aes(x=log1p(Fare), y=Age, color=Survived)) +
        geom_point() +
        facet_grid(Sex ~.)

    png('fare_age_sex.png', width=1200, height=800)
    print(p)
    dev.off()
}

plot.data()