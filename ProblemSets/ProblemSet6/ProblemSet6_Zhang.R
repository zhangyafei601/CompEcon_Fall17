# Call in libraries
library(haven)
library(optimx)
library(foreign)
library(sandwich)
library(AER)
library(stargazer)

# Read in data
ps6_data <- read_dta("C:/Users/yafei/Dropbox/sponsor/data and program/ds_fac_underp.dta") 

# keep only columns will use
ps6_df <- ps6_data[,c("spread_adj_par","borrower_lead_amt","amount", "term", "dum_public", "dum_sponsor", "dum_prepay", 
                      "dum_floor", "dum_refin", "dum_secure", "dum_cov", "rate_blr_sp", "primarypurpose", "sic2",
                      "loan_year", "companyid", "newtreat")]

summary(ps6_df)

# Calculate variables
ps6_df['abs_spread_adj'] <- abs(ps6_df['spread_adj_par'])
ps6_df['log_amt'] <- log(ps6_df['amount'])
ps6_df['log_term'] <- log(ps6_df['term'])


# Generate dummy variable for fixed effects
ps6_df$purpg <- factor(ps6_df$primarypurpose)
ps6_df$indg <- factor(ps6_df$sic2)
ps6_df$yearg <- factor(ps6_df$loan_year)
ps6_df$rateg <- factor(ps6_df$rate_blr_sp)
ps6_df$companyidg <- factor(ps6_df$companyid)


ps6_df <- ps6_df[!(is.na(ps6_df$abs_spread_adj) | ps6_df$abs_spread_adj==""), ]

# OLS without fixed effects
adj_OLS <- lm(abs_spread_adj ~ borrower_lead_amt + log_amt + log_term + dum_public + 
                dum_sponsor + dum_prepay + dum_floor + dum_refin + dum_secure + dum_cov, data = ps6_df)
summary(adj_OLS)


# OLS with fixed effects
adj_OLS_fix <- lm(abs_spread_adj ~ borrower_lead_amt + log_amt + log_term + dum_public + 
                    dum_sponsor + dum_prepay + dum_floor + dum_refin + dum_secure + dum_cov + 
                    purpg + indg + yearg + rateg + companyidg, data = ps6_df)
# Return robust standard errors
coeftest(adj_OLS_fix, vcov = vcovHC(adj_OLS_fix, "HC1"))


# IV model
# treat is the instrument variable in the first stage regression
adj_IV <- ivreg(abs_spread_adj ~ borrower_lead_amt + log_amt + log_term + dum_public + 
                  dum_sponsor + dum_prepay + dum_floor + dum_refin + dum_secure + dum_cov + 
                  purpg + indg + yearg + rateg + companyidg |
                  newtreat + log_amt + log_term + dum_public + dum_sponsor + dum_prepay + dum_floor + 
                  dum_refin + dum_secure + dum_cov + purpg + indg + yearg + rateg + companyidg, data = ps6_df)
summary(adj_IV)

# Output tables
stargazer(adj_OLS, adj_OLS_fix, adj_IV, type = 'text', title="PS6 Regression Results", 
          align=TRUE, omit = c('purpg', 'indg', 'yearg', 'rateg', 'companyidg'),
          dep.var.caption = "Dependant Variable: Spread Adjustments",
          dep.var.labels = "",
          omit.stat = c("ser", "f", "rsq"),
          add.lines = list(c("Year", "No", "Yes", "Yes"), c("Industry", "No", "Yes", "Yes"),
                           c("Lead Arranger", "No", "Yes", "Yes"), c("S&P Ratings", "No", "Yes", "Yes"),
                           c("Loan Purpose", "No", "Yes", "Yes")),
          notes = ("Standard errors are reported below the coefficients"),
          no.space = TRUE,
          out = "C:/Users/yafei/CompEcon_Fall17/R/models.txt")
