# Student t option pricing
Simulates option pricing when the terminal stock price follows a normal or Student t distribution rather than the usual lognormal. The distribution of stock market gross return, 
`S(t)/S(t-k)`, from 1 day through at least 3 months is closer to Student t than lognormal empirically, because it does not have positive skew and has excess kurtosis. This may explain the implied volatility smile (varying implied volatility by strike) when the Black-Scholes model is used. The Student t distribution approaches the normal for large df. Run with `python xdof_option.py`.

![Alt text](/smile.png)

```
Simulation parameters:
                  parameter    value
                        dfs   10, 30
             include_normal     True
          include_lognormal     True
                    n_paths 10000000
              mean_terminal    100.0
               std_terminal     10.0
                       spot    100.0
                       rate      0.0
           time_to_maturity      1.0
               strike_start     80.0
                 strike_end    120.0
                strike_step      5.0
plot_terminal_distributions    False
                       seed    12345
Distribution summary:
   distribution       mean       std      skew  kurtosis       min        max
Student-t df=10 100.000486 10.002479  0.000951  0.997610  0.000000 241.662530
Student-t df=30  99.992272  9.998831 -0.001024  0.234876 35.470006 162.801259
         Normal 100.002427  9.999116  0.000150  0.000442 48.072628 150.389872
      Lognormal 100.001150 10.000080  0.300587  0.162487 60.670338 164.729625
```
