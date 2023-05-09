import numpy as np
import pandas as pd
import igraph as ig
import matplotlib.pyplot as plt

from copy import deepcopy
from statsmodels.stats.multitest import fdrcorrection


def get_list_shape(l):
    shape = []
    for interior_list in l:
        shape.append(len(interior_list))
    return shape


def flatten_list(l):
    flattened = []
    for interior_list in l:
        flattened += deepcopy(interior_list)
    return flattened


def reshape_list(flat_list, shape):
    deepcopy_flat_list = deepcopy(flat_list)
    original_list = []
    for list_length in shape:
        original_list.append([])
        for _ in range(list_length):
            original_list[-1].append(deepcopy_flat_list.pop(0))
    return original_list


def bidimensional_l_mean(l):
    mean = []
    for interior_list in l:
        if len(interior_list) > 0:
            mean.append(sum(interior_list) / len(interior_list))
        else:
            mean.append(0)
    return mean


class Node:

    def __init__(self, data, current_attribute_value=None, parent=None, p_value=None, fdr_corrected_value=None):
        
        self.data = data
        self.current_attribute_value = current_attribute_value
        self.parent = parent
        self.p_value = p_value
        self.fdr_corrected_value = fdr_corrected_value

        self.next_request = None
        self.children = []
    

    def insert_children(self, applied_request, children_data_regions, children_attribute_values, children_p_values, children_fdr_corrected_values):
        self.next_request = applied_request
            
        for child_data_region, child_attribute_value, child_p_value, child_fdr_corrected_value in zip(children_data_regions, children_attribute_values, children_p_values, children_fdr_corrected_values):
            self.children.append(
                Node(
                    data=child_data_region,
                    current_attribute_value=child_attribute_value,
                    parent=self,
                    p_value=child_p_value,
                    fdr_corrected_value=child_fdr_corrected_value
                )
            )


    def get_attribute_values(self, show_next_request_attribute=False, reverse=False):
        attribute_values = []

        if self.current_attribute_value is not None:
            attribute_values.append(self.current_attribute_value)

        if show_next_request_attribute:
            if self.next_request is not None:
                attribute_values.append(self.next_request.attribute_value)
        
        ancestor = self.parent
        while ancestor is not None:
            if ancestor.current_attribute_value is not None:
                attribute_values.append(ancestor.current_attribute_value)
            attribute_values.append(ancestor.next_request.attribute_value)
            ancestor = ancestor.parent

        if reverse:
            attribute_values = list(reversed(attribute_values))

        return attribute_values
    
    
    def get_single_choice_attributes(self):
        single_choice_attributes = []

        for attribute_value in self.get_attribute_values():
            if attribute_value.attribute.single_choice:
                single_choice_attributes.append(attribute_value.attribute)
        
        return single_choice_attributes
    

    def get_available_attributes(self, all_attributes):
        available_attributes = []

        single_choice_attributes = self.get_single_choice_attributes()
        for attribute in all_attributes:
            if attribute not in single_choice_attributes:
                available_attributes.append(attribute)
        
        return available_attributes
    

    def get_available_attribute_values(self, all_attributes):
        available_attribute_values = []
        
        not_available_attribute_values = self.get_attribute_values()
        availabe_attributes = self.get_available_attributes(all_attributes)
        for attribute in availabe_attributes:
            for attribute_value in attribute.values:
                if attribute_value not in not_available_attribute_values:
                    available_attribute_values.append(attribute_value)
        
        return available_attribute_values
    

    def get_available_requests(self, all_attributes, all_requests):
        available_requests = []

        available_attribute_values = self.get_available_attribute_values(all_attributes)
        for request in all_requests:
            if request.attribute_value in available_attribute_values:
                available_requests.append(request)
        
        return available_requests


class TreeBuilder:
    
    def __init__(self, possible_requests, possible_attributes, alpha=0.05):
        self.possible_requests = possible_requests
        self.possible_attributes = possible_attributes
        self.alpha = alpha
    

    def select_and_apply_next_request(self, node):
        requests, p_values, next_data_regions, next_attribute_values = [], [], [], []

        for request in node.get_available_requests(self.possible_attributes, self.possible_requests):
            request_p_values, request_next_data_regions, request_next_attribute_values = request.apply(node, self.possible_attributes, self.alpha)
            if len(request_p_values) > 0:
                requests.append(request)
                p_values.append(request_p_values)
                next_data_regions.append(request_next_data_regions)
                next_attribute_values.append(request_next_attribute_values)
        
        fdr_corrected = fdrcorrection(flatten_list(p_values), alpha=self.alpha, method='n')
        fdr_vals = reshape_list(fdr_corrected[1].tolist(), get_list_shape(p_values))
        fdr_mean = bidimensional_l_mean(fdr_vals)
        argmin = np.argmin(fdr_mean)

        applied_request = requests[argmin]
        children_data_regions = next_data_regions[argmin]
        children_attribute_values = next_attribute_values[argmin]
        children_p_values = p_values[argmin]
        children_fdr_corrected_values = fdr_vals[argmin]
        
        node.insert_children(applied_request, children_data_regions, children_attribute_values, children_p_values, children_fdr_corrected_values)
    

    def build_tree(self, root, n):
        if n > 0:
            self.select_and_apply_next_request(root)
            for child in root.children:
                self.build_tree(child, n-1)


class ResultsDataFrameBuilder:
    
    def __init__(self, root):
        self.root = root

    def get_power(self, node):
        return 0

    def get_fdr(self, node):
        if len(node.children) > 0:
            return sum([child.fdr_corrected_value for child in node.children]) / len(node.children)
        else:
            return 0

    def get_reward(self, node):
        return 0

    def get_max_pval(self, node):
        if len(node.children) > 0:
            return max([child.p_value for child in node.children])
        else:
            return 0

    def get_min_pval(self, node):
        if len(node.children) > 0:
            return min([child.p_value for child in node.children])
        else:
            return 0

    def get_sum_pval(self, node):
        if len(node.children) > 0:
            return sum([child.p_value for child in node.children])
        else:
            return 0

    def get_coverage(self, node):
        if len(node.data['cust_id'].tolist()) > 0:
            return len(set(node.data['cust_id'].tolist())) / len(node.data['cust_id'].tolist())
        else:
            return 0

    def get_episodes(self, node):
        return 0

    def get_steps_in_episode(self, node):
        return 0

    def get_input_data_region(self, node):
        return '_'.join([att_val.value for att_val in node.get_attribute_values(show_next_request_attribute=True, reverse=True)])

    def get_attributes_combination_input_data_region(self, node):
        return '_'.join([att_val.attribute.attribute for att_val in node.get_attribute_values(show_next_request_attribute=True, reverse=True)])

    def get_cust_ids_input_data_region(self, node):
        return set(node.data['cust_id'].tolist())

    def get_hypothesis(self, node):
        if node.next_request is not None:
            return ['One-Sample', node.next_request.test.null_value, node.next_request.test.aggregation]
        else:
            return ['', '', '']

    def get_action(self, node):
        return 'Exploit'

    def get_size_output_set(self, node):
        return len(node.children)

    def get_output_data_regions(self, node):
        output_data_regions = []
        for child in node.children:
            output_data_regions.append('_'.join([att_val.value for att_val in child.get_attribute_values(reverse=True)]))
        return output_data_regions

    def get_attributes_combination_output_data_regions(self, node):
        attributes_combination_output_data_regions = []
        for child in node.children:
            attributes_combination_output_data_regions.append('_'.join([att_val.attribute.attribute for att_val in child.get_attribute_values(reverse=True)]))
        return attributes_combination_output_data_regions

    def get_cust_ids_output_data_regions(self, node):
        cust_ids_output_data_regions = []
        for child in node.children:
            cust_ids_output_data_regions.append(set(child.data['cust_id'].tolist()))
        return cust_ids_output_data_regions

    def get_size_ouptput_data_regions(self, node):
        size_ouptput_data_regions = []
        for child in node.children:
            size_ouptput_data_regions.append(len(child.data))
        return size_ouptput_data_regions
    
    def build_dataframe(self):
        power, fdr, reward, max_pval, min_pval, sum_pval = [], [], [], [], [], []
        coverage, episodes, steps_in_episode, input_data_region = [], [], [], []
        attributes_combination_input_data_region, cust_ids_input_data_region = [], []
        hypothesis, action, size_output_set, output_data_regions = [], [], [], []
        attributes_combination_output_data_regions = []
        cust_ids_output_data_regions, size_ouptput_data_regions = [], []

        current_node_index = 0
        current_level = [(None, self.root)]
        while current_level:
            next_level = []
            for parent_index, node in current_level:
                if node.next_request is not None:
                    power.append(self.get_power(node))
                    fdr.append(self.get_fdr(node))
                    reward.append(self.get_reward(node))
                    max_pval.append(self.get_max_pval(node))
                    min_pval.append(self.get_min_pval(node))
                    sum_pval.append(self.get_sum_pval(node))
                    coverage.append(self.get_coverage(node))
                    episodes.append(self.get_episodes(node))
                    steps_in_episode.append(self.get_steps_in_episode(node))
                    input_data_region.append(self.get_input_data_region(node))
                    attributes_combination_input_data_region.append(self.get_attributes_combination_input_data_region(node))
                    cust_ids_input_data_region.append(self.get_cust_ids_input_data_region(node))
                    hypothesis.append(self.get_hypothesis(node))
                    action.append(self.get_action(node))
                    size_output_set.append(self.get_size_output_set(node))
                    output_data_regions.append(self.get_output_data_regions(node))
                    attributes_combination_output_data_regions.append(self.get_attributes_combination_output_data_regions(node))
                    cust_ids_output_data_regions.append(self.get_cust_ids_output_data_regions(node))
                    size_ouptput_data_regions.append(self.get_size_ouptput_data_regions(node))
                for child in node.children:
                    next_level.append((current_node_index, child))
                current_node_index += 1
            current_level = next_level

        return pd.DataFrame({
            'power': power,
            'fdr': fdr,
            'reward': reward,
            'max_pval': max_pval,
            'min_pval': min_pval,
            'sum_pval': sum_pval,
            'coverage': coverage,
            'episodes': episodes,
            'steps_in_episode': steps_in_episode,
            'input_data_region': input_data_region,
            'attributes_combination_input_data_region': attributes_combination_input_data_region,
            'cust_ids_input_data_region': cust_ids_input_data_region,
            'hypothesis': hypothesis,
            'action': action,
            'size_output_set': size_output_set,
            'output_data_regions': output_data_regions,
            'attributes_combination_output_data_regions': attributes_combination_output_data_regions,
            'cust_ids_output_data_regions': cust_ids_output_data_regions,
            'size_ouptput_data_regions': size_ouptput_data_regions,
        })


def plot_tree(root):
    edge_list = []
    edge_labels = []
    vertex_labels = []
    
    current_node_index = 0
    current_level = [(None, root)]
    while current_level:
        next_level = []
        for parent_index, node in current_level:
            if node.next_request is not None:
                vertex_labels.append(str(node.next_request))
            else:
                vertex_labels.append('_'.join([att_val.value for att_val in node.get_attribute_values(show_next_request_attribute=True, reverse=True)]))
            
            if parent_index is not None:
                edge_labels.append(node.current_attribute_value.value)
                edge_list.append((parent_index, current_node_index))
                
            for child in node.children:
                next_level.append((current_node_index, child))
            current_node_index += 1
        current_level = next_level
    
    g = ig.Graph(
        edge_list,
        directed=True,
    )

    # fig, ax = plt.subplots(figsize=(32,24))

    fig, ax = plt.subplots()

    ig.plot(
        g,
        target=ax,
        layout="tree",
        vertex_label=vertex_labels,
        edge_label=edge_labels,
        edge_background="white",
        vertex_color="lightblue",
    )
