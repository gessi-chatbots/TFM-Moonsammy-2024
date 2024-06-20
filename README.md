# Unveiling Competition Dynamics in Mobile App Markets through User Review

This replication package contains the software resources and data artefacts used for preliminary experimentation and analysis of early research results. 

## Structure

This package is structured in 3 main folders:

- **data** - Contains the data set of reviews used for experimentation.
- **scripts** - Contains the software files to apply the automatic processes depicted in our approach, mainly (1) the extraction of metrics, (2) the computation of statistical metrics, and (3) the selection of reviews for summarization.
- **results** - Contains the output files generated by the previous automated processes (1-3).

## Dataset

We selected a subset of microblogging apps which included Twitter, Mastodon, and 10 additional microblogging Android mobile apps based on user crowdsourced software recommendations from the [AlternativeTo platform](https://alternativeto.net/software/twitter/?platform=android). For each app, we collected all reviews available in multiple repositories published within a time window of 52 weeks (~1 year), from June 9th, 2022 to June 7th, 2023 (included). From the complete list of 12 microblogging apps, we excluded 2 apps for which the number of available reviews was insufficient for statistical analysis. The complete list of apps and the number of available reviews is reported in the following table.

| App name      | App package                 | #reviews |
|---------------|-----------------------------|----------|
| X (Twitter)   | com.twitter.android         | 168,335  |
| Truth Social  | com.truthsocial.android.app | 4,580    |
| VK            | com.vkontakte.android       | 3,677    |
| Hive Social   | org.hiveinc.TheHive.android | 3,054    |
| GETTR         | com.gettr.gettr             | 2,592    |
| MeWe          | com.mewe                    | 2,660    |
| Mastodon      | org.joinmastodon.android    | 1,441    |
| Bluesky       | xyz.blueskyweb.app          | 625      |
| Minds         | com.minds.mobile            | 423      |
| CounterSocial | counter.social.android      | 252      |

The complete data set of reviews is available at ```data/microblogging-review-set.json```.

## Experimentation

In this section, we provide a detailed step-by-step description of the experimentation process using the scripts available in the ```scripts``` folder.

- Install the Python module requirements:
    
    ```pip install -r scripts/requirements.txt```

### Metric computation
    
- Compute the **review count (c)** metric. By default, we use a time window of size ```w = 7``` and an origin date ```tΩ = Jun 09, 2022``` (this applies to all metrics).

    ```python ./scripts/compute_review_count.py -i ./data/microblogging-review-set.json -w 7 -t 'Jun 09, 2022' -o ./results/metrics```
    
- Compute the **review rating (r)** metric. 

    ```python ./scripts/compute_review_rating.py -i ./data/microblogging-review-set.json -w 7 -t 'Jun 09, 2022' -o ./results/metrics```
    
- Compute the **review polarity (p)** metric. To run this script, we build on the work of a sentiment analysis service, available in a [GitHub repository](https://github.com/AgustiGM/sa_filter_tool). Therefore, it is necessary to install and run the service as depicted in the [README file](https://github.com/AgustiGM/sa_filter_tool#readme).

	Once the service is up and running, the review polarity metric is ready to be computed. **[NOTE: This process might take a while]**.

    ```python ./scripts/compute_review_polarity.py -i ./data/microblogging-review-set.json -w 7 -t 'Jun 09, 2022' -o ./results/metrics```

### Event visualization

After metric extraction, events can be visualized in isolation for an event-based analysis.
    
- Collect the output files and paste the data into the ```results/metrics/event_monitoring.xlsx``` spreadsheet. This spreadsheet is configured to provide automated visualization of events, for which a sensitivity factor ```k = 2``` is set as default (can be modified). Specifically:
	- 	```results/review_count.csv``` at ```review_count!A2:BA12```
	- 	```results/review_rating.csv``` at ```review_rating!A2:BA12```
	- 	```results/review_polarity.csv``` at ```review_polarity!A2:BA12```

- A heat-map containing the visualization of reported events for the computed metrics is available in ```results/metrics/event_monitoring.xlsx```, under the sheet named ```event_monitoring```.

- For generating the files for future analysis, it is necessary to retrieve the events in .csv format

```python ./scripts/compute_events.py -i results/metrics/event_monitoring.xlsx -o results/correlation -k 2```

### Correlation analysis

Event-based metrics can then be used to compute review-based metric correlation.

- Compute correlation.

```python ./scripts/compute_correlation.py -i results/metrics/event_monitoring.xlsx -w 1 -o results/correlation/clusters.csv```

### Potentially correlated events

Intersection of events and correlated periods is used to retrieve potentially correlated events.

- Compute intersection

```python ./scripts/event_correlation_intersection.py -e results/correlation -c results/correlation/clusters.csv -o results/intersection.csv```

### Summarization

To conduct a sample summarization of a selected time window:

- Use any reported event to extract a sample representation of the reviews published for a given app within a time window. The script requires specification of the following parameters: the app package (```-a```), the date interval (```-d```), the output folder (```-o```) and the number of reviews to collect in the sample (```-n```). 
	
    The folowing example collects a sample of 50 reviews from the Twitter app published between October 27th, 2022 and November 2nd, 2022 (i.e., the week when Twitter formally announced the buyout from Musk).

	```python ./scripts/get_reviews_by_date.py -i ./data/microblogging-review-set.json -a com.twitter.android -d 'Oct 27, 2022 - Nov 02, 2022' -o results -n 50```

- An output file containing the sample set of reviews is generated using the name template ```<a>-reviews-<d>.csv``` where ```<a>``` is the app package and ```d``` is the date interval. For instance, for the previous example, the following file will be generated:

	```com.twitter.android-reviews-Oct 27, 2022 - Nov 02, 2022```
    
- Use ChatGPT to request a summarization of the most relevant events highlighted in the sample set of reviews. To do so, we used a prompt engineering approach within a zero shot learning context, for which the following prompt was designed:

	```
    <review-set>
    
    Identify and summarize the most significant event raised by this set of reviews extracted from mobile app repositories.
    ```
    
    Where ```<review-set>``` is the content of the sample review file generated in step #7.
    
    Files ```example_N.txt``` are the complete summarized output provided by ChatGPT from the Examples 1 -> 6 reported in the original manuscript.
    
	(!) Please note that systematic repetitions of this process can lead to slightly different results, mainly because of (1) the random selection of a sample set of reviews, and (2) the internal behaviour of ChatGPT.

To conduct the automatic generation of multiple sets of reviews based on potentially correlated events:

- Run a script to generate the batch of reviews to be used as input for ChatGPT prompt:

```python scripts/run_multiple_get_review_by_date.py -i results/correlation/intersection.csv -o results/correlation/reviews```
