customers <- read.csv("/home/alex/projects/customers.csv")
# TO BE DONE: TODO pick k properly
fit <- kmeans(customers[, c("spend", "visits")], centers = 4)
