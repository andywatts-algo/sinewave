## TODO
* Model is learning, but not great.   Move onto 0DTE.  Come back and test feature engineering


## Train/Test #v0.2
* Data
    * Random frequency and amplitude sinewave
* Training
    * 850 ep_rew_mean after 75k episodes
* Testing
* Conclusion





## Train/Test #v0.1
* Data
    * 1000 rows
    * 10 frequency
* Environment   
    * Obs: Last 3 days price change   
    * Action: Up|Down   
    * Reward: +1 if correct; -1.5 if incorrect   
* Training
    * 978 ep_rew_mean after 1M episodes
* Testing
    * Same amplitude and frequency, but different start? WORKS 94%
    * Different frquency?
        * 20 = 87%; 30 = 80%; 50 = 65%
    * Different amplitude?
        * 4-7 = 82%
* Conclusion
    * Not very transferable from fixed frequency/amplitude.
    * Retrain with variable frequency/amplitude







