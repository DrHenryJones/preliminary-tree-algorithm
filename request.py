import datetime
import random
import copy
import statistics
import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from scipy import stats


def one_sample_hypothesis_test(data, aggregation, null_value, alternative):
    """
    Statistical hypothesis testing.

    This function applies a one-sample statistical
    test to draw a conclusion about a population
    parameter or distribution.

    Parameters
    ----------
    data : array
        Sample observation.
    aggregation : {'mean', 'variance', 'distribution'}
        Aggregation function.
        > 'mean' : applies a one-sample t-test
        > 'variance' : applies a one-sample variance Chi-square test
        > 'distribution' : applies a one-sample Kolmogorov-Smirnov test
    null_value : float or {'norm', 'expon'}
        Null value of the hypothesis. Is str if aggregation is 'distribution', otherwise is float.
        > 'norm': gaussian distribution
        > 'expon': exponential distribution
    alternative : {'less', 'greater', 'two-sided'}
        Defines the alternative hypothesis.
        > 'less': mean of sample is less than the null value
        > 'greater': mean of sample is greater than null value
        > 'two-sided': mean of sample is different than null value

    Returns
    -------
    float
        The p-value associated with the given alternative.
    """
    if aggregation == 'mean':
        return stats.ttest_1samp(data, popmean=null_value, alternative=alternative).pvalue
    
    elif aggregation == 'variance':
        n = len(data)
        test_statistic = (n - 1) * np.var(data) / null_value
        if alternative == 'less':
            return stats.chi2.cdf(test_statistic, n - 1)
        elif alternative == 'greater':
            return stats.chi2.sf(test_statistic, n - 1)
        elif alternative == 'two-sided':
            return 2 * min(stats.chi2.cdf(test_statistic, n - 1), stats.chi2.sf(test_statistic, n - 1))
        else:
            raise ValueError("alternative must be 'less', 'greater' or 'two-sided'")
    
    elif aggregation == 'distribution':
        return stats.kstest(data, null_value, alternative=alternative).pvalue
    
    else:
        raise ValueError("aggregation must be 'mean', 'variance' or 'distribution'")


class OneSampleStatisticalTest:
    
    def __init__(self, aggregation, null_value, alternative):
        self.aggregation = aggregation
        self.null_value = null_value
        self.alternative = alternative
    

    def get_p_value(self, data):
        return one_sample_hypothesis_test(data, self.aggregation, self.null_value, self.alternative)


class AttributeValue:
    
    def __init__(self, value, attribute):
        self.value = value
        self.attribute = attribute
    

    def filter_data_region(self, data):
        if self.attribute.single_choice:
            next_data_region = data[data[self.attribute.attribute] == self.value]
        else:
            next_data_region = data[data[self.attribute.attribute].str.contains(self.value)]
        return next_data_region
    
    
    def __str__(self):
        return self.value


class Attribute:
    
    def __init__(self, attribute, values, single_choice, item_attribute):
        self.attribute = attribute
        self.single_choice = single_choice
        self.item_attribute = item_attribute

        self.values = []
        for value in values:
            self.values.append(AttributeValue(value, self))
    

    def __str__(self):
        return self.attribute


class AbstractRequest(ABC):

    def __init__(self, attribute_value, test):
        self.attribute_value = attribute_value
        self.test = test
    

    @abstractmethod
    def apply(self):
        pass


class MovieLensOneSampleRequest(AbstractRequest):

    def apply(self, node, all_attributes, alpha):
        p_values, next_data_regions, next_attribute_values = [], [], []

        top_data_region = self.attribute_value.filter_data_region(node.data)

        for attribute_value in node.get_available_attribute_values(all_attributes):
            if attribute_value != self.attribute_value:

                msr_data_region = attribute_value.filter_data_region(top_data_region)

                if attribute_value.attribute.item_attribute:
                        # next_data_region = top_data_region[top_data_region['cust_id'].isin(msr_data_region['cust_id'].to_list())]
                        next_data_region = msr_data_region
                else:
                    next_data_region = msr_data_region
                
                if len(next_data_region) > 0:
                    p_value = self.test.get_p_value(msr_data_region['rating'].to_numpy())
                    if p_value < alpha:
                        p_values.append(p_value)
                        next_data_regions.append(next_data_region)
                        next_attribute_values.append(attribute_value)
        
        return p_values, next_data_regions, next_attribute_values
    
    
    def __str__(self):
        if self.attribute_value.attribute.item_attribute:
            request_str = 'groups whose rating ' + self.test.aggregation + ' for ' + self.attribute_value.value + ' movies '
        else:
            request_str = self.attribute_value.value + ' groups whose rating ' + self.test.aggregation + ' '
        
        request_str += '\n'
        
        if self.test.aggregation == 'distribution':
            request_str += 'does not follow a ' + self.test.null_value + ' distribution'
        else:
            if self.test.alternative == 'two-sided':
                request_str += 'is different than ' + str(self.test.null_value)
            elif self.test.alternative == 'less':
                request_str += 'is less than ' + str(self.test.null_value)
            elif self.test.alternative == 'greater':
                request_str += 'is greater than ' + str(self.test.null_value)
        return request_str.capitalize()
