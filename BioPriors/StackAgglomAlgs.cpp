#include <cilk/cilk.h>
#include <cilk/cilk_api.h>

#include "../Algorithms/MergePriorityFunction.h"
#include "StackAgglomAlgs.h"
#include "../Stack/Stack.h"
#include "MitoTypeProperty.h"
#include "../Algorithms/FeatureJoinAlgs.h"
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread/mutex.hpp>

#include <algorithm>

#include <vector>

#include <climits>

using std::vector;

namespace NeuroProof {

bool is_mito(RagNode_t* rag_node)
{
    MitoTypeProperty mtype;
    try {
        mtype = rag_node->get_property<MitoTypeProperty>("mito-type");
    } catch (ErrMsg& msg) {
    }

    if ((mtype.get_node_type()==2)) {	
        return true;
    }
    return false;
}



void agglomerate_stack(Stack& stack, double threshold,
                        bool use_mito, bool use_edge_weight, bool synapse_mode)
{
    if (threshold == 0.0) {
        return;
    }

    RagPtr rag = stack.get_rag();
    FeatureMgrPtr feature_mgr = stack.get_feature_manager();

    MergePriority* priority = new ProbPriority(feature_mgr.get(), rag.get(), synapse_mode);
    boost::posix_time::ptime start = boost::posix_time::microsec_clock::local_time();
    priority->initialize_priority(threshold, use_edge_weight);
    boost::posix_time::ptime now = boost::posix_time::microsec_clock::local_time();
    cout << endl << "------------------------ INIT PRIORITY Q: " << (now - start).total_milliseconds() << " ms\n" << endl;
    DelayedPriorityCombine node_combine_alg(feature_mgr.get(), rag.get(), priority); 


    while (!(priority->empty())) {

        RagEdge_t* rag_edge = priority->get_top_edge();

        if (!rag_edge) {
            continue;
        }

        RagNode_t* rag_node1 = rag_edge->get_node1();
        RagNode_t* rag_node2 = rag_edge->get_node2();

        if (use_mito) {
            if (is_mito(rag_node1) || is_mito(rag_node2)) {
                continue;
            }
        }

        Node_t node1 = rag_node1->get_node_id(); 
        Node_t node2 = rag_node2->get_node_id();
        
        // retain node1 
        stack.merge_labels(node2, node1, &node_combine_alg);
    }

    delete priority;
}


inline void remove(vector<RagEdge_t*> & v, RagEdge_t* item) {
    v.erase(std::remove(v.begin(), v.end(), item), v.end());
}


inline Node_t get_actual_label (Node_t label, map<Node_t, Node_t> &map) {
    while (label != map[label])
        label = map[label];

    return label;
}



void agglomerate_stack_parallel(Stack& stack, int num_top_edges, double threshold, bool use_mito, bool use_edge_weight, bool synapse_mode) {
    if (threshold == 0.0) {
        return;
    }    

    RagPtr rag = stack.get_rag();
    FeatureMgrPtr feature_mgr = stack.get_feature_manager();


    MergePriority* priority = new ProbPriority(feature_mgr.get(), rag.get(), synapse_mode);
    boost::posix_time::ptime start = boost::posix_time::microsec_clock::local_time();
    priority->initialize_priority(threshold, use_edge_weight);
    boost::posix_time::ptime now = boost::posix_time::microsec_clock::local_time();
    cout << endl << "------------------------ INIT PRIORITY Q: " << (now - start).total_milliseconds() << " ms" << endl;
    DelayedPriorityCombine node_combine_alg(feature_mgr.get(), rag.get(), priority);

    boost::posix_time::ptime t1 = boost::posix_time::microsec_clock::local_time();
    int loop_count = 0;
    while (!(priority->empty())) {
        loop_count++;
        vector<RagEdge_t*> edges_to_process = priority->get_top_independent_edges(num_top_edges);
        for (vector<RagEdge_t*>::iterator it = edges_to_process.begin(); it != edges_to_process.end(); ++it) {
            RagNode_t* rag_node1 = (*it)->get_node1();
            RagNode_t* rag_node2 = (*it)->get_node2();

            // if (use_mito) {
            //     if (is_mito(rag_node1) || is_mito(rag_node2)) {
            //         continue;
            //     }
            // }

            Node_t node1 = rag_node1->get_node_id(); 
            Node_t node2 = rag_node2->get_node_id();
            
            // retain node1 
            stack.merge_labels(node2, node1, &node_combine_alg);
        }
    }
    boost::posix_time::ptime t2 = boost::posix_time::microsec_clock::local_time();
    cout << "------------------------ TIME AGGLO LOOP: " << (t2 - t1).total_milliseconds() << " ms" << endl;

    // cout << "------------------------ TIME TAKING OUT EDGES: " << total_time_take_out/1000000 << " ms" << endl;
    // cout << "------------------------ TIME MERGING NODES: " << total_time_merge/1000000 << " ms" << endl;
    // cout << "------------------------ TIME LOOP OVERHEAD: " << loop_time/1000000 << " ms" << endl;
    // cout << "AGGLO LOOP COUNT: " << loop_count << endl;
    
    delete priority;
}

/*
void agglomerate_stack_parallel(Stack& stack, double threshold, bool use_mito, bool use_edge_weight, bool synapse_mode) {
    // agglomerate_stack(stack, threshold, use_mito, use_edge_weight, synapse_mode);

    if (threshold == 0.0) {
        return;
    }

    RagPtr rag = stack.get_rag();
    FeatureMgrPtr feature_mgr = stack.get_feature_manager();


    MergePriority* priority = new ProbPriority(feature_mgr.get(), rag.get(), synapse_mode);
    boost::posix_time::ptime start = boost::posix_time::microsec_clock::local_time();
    priority->initialize_priority(threshold, use_edge_weight);
    boost::posix_time::ptime now = boost::posix_time::microsec_clock::local_time();
    cout << endl << "------------------------ INIT PRIORITY Q: " << (now - start).total_milliseconds() << " ms\n";
    DelayedPriorityCombine node_combine_alg(feature_mgr.get(), rag.get(), priority);

    int num_top_edges = 8;

    while (!(priority->empty())) {

        // Round 1
        vector<RagEdge_t*> rag_edge_vec;
        int max_labels = 100000;
        int node_priority_update_array [max_labels];
        // int hi = 1;
        // __sync_bool_compare_and_swap(&hi, 1, 0);
        for (int i = 0; i < num_top_edges;) {
            if (priority->empty())
                break;
            RagEdge_t* rag_edge = priority->get_top_edge();
            if (!rag_edge) {
                continue;
            }

            rag_edge_vec.push_back(rag_edge);
            i++;
        }


        if (rag_edge_vec.size() == 0)
            continue;
        
        // Round 2
        map<Node_t, Node_t> node_look_up;
        int num_edges_processed = 0;
        while (num_edges_processed < num_top_edges) {
            vector <RagEdge_t*> edges_to_process;

            for (int i = 0; i < max_labels; ++i) {
                node_priority_update_array[i] = num_top_edges + 1;
            }

            for (int i = 0; i < rag_edge_vec.size(); ++i) {
                int index1 = rag_edge_vec[i]->get_node1()->get_node_id();
                int index2 = rag_edge_vec[i]->get_node2()->get_node_id();

                if (node_priority_update_array[index1] > i)
                    node_priority_update_array[index1] = i;
                if (node_priority_update_array[index2] > i)
                    node_priority_update_array[index2] = i;
            }

            for (vector<RagEdge_t*>::iterator it = rag_edge_vec.begin(); it != rag_edge_vec.end(); ++it) {
                int index1 = (*it)->get_node1()->get_node_id();
                int index2 = (*it)->get_node2()->get_node_id();
                Node_t actual_index1 = get_actual_label(index1, node_look_up);
                Node_t actual_index2 = get_actual_label(index2, node_look_up);
                if (node_priority_update_array[index1] == node_priority_update_array[index2]) {
                    if (node_look_up.find(index1) == node_look_up.end())
                        node_look_up[index1] = index1;
                    if (node_look_up.find(index2) == node_look_up.end())
                        node_look_up[index2] = index2;
                    edges_to_process.push_back(*it);
                    num_edges_processed++;
                }
            }
            
            for (vector<RagEdge_t*>::iterator it = edges_to_process.begin(); it != edges_to_process.end(); ++it) {
                Node_t index1 = (*it)->get_node1()->get_node_id();
                Node_t index2 = (*it)->get_node2()->get_node_id();
                Node_t actual_index1 = get_actual_label(index1, node_look_up);
                Node_t actual_index2 = get_actual_label(index2, node_look_up);
                node_look_up[actual_index2] = actual_index1;
                if (actual_index1 != actual_index2) {
                    stack.merge_labels(actual_index2, actual_index1, &node_combine_alg);
                }
                remove(rag_edge_vec, *it);
            }
        }
        
        // // Round 3
        // for (map<Node_t, Node_t>::iterator it = node_look_up.begin(); it != node_look_up.end(); ++it) {
        //     if (it->first != it->second) {
        //         // retain  it->second
        //         stack.merge_labels(it->first, it->second, &node_combine_alg);
        //     }
        // }


        // RagEdge_t* rag_edge = rag_edge_vec[0];
        // // RagEdge_t* rag_edge = priority->get_top_edge();

        // if (!rag_edge) {
        //     continue;
        // }

        // RagNode_t* rag_node1 = rag_edge->get_node1();
        // RagNode_t* rag_node2 = rag_edge->get_node2();

        // if (use_mito) {
        //     if (is_mito(rag_node1) || is_mito(rag_node2)) {
        //         continue;
        //     }
        // }

        // Node_t node1 = rag_node1->get_node_id(); 
        // Node_t node2 = rag_node2->get_node_id();
        
        // // retain node1 
        // stack.merge_labels(node2, node1, &node_combine_alg);
    }

    delete priority;    
}
*/


void agglomerate_stack_mrf(Stack& stack, double threshold, bool use_mito)
{
    if (threshold == 0.0) {
        return;
    }

    boost::posix_time::ptime start = boost::posix_time::microsec_clock::local_time();
    agglomerate_stack_parallel(stack, 0.06, use_mito); //0.01 for 250, 0.02 for 500
    boost::posix_time::ptime now = boost::posix_time::microsec_clock::local_time();
    cout << endl << "------------------------ AGGLO FIRST PASS: " << (now - start).total_milliseconds() << " ms\n";
    stack.remove_inclusions();	  	
    cout <<  "Remaining regions: " << stack.get_num_labels();	

    RagPtr rag = stack.get_rag();
    FeatureMgrPtr feature_mgr = stack.get_feature_manager();

    unsigned int edgeCount=0;	

    start = boost::posix_time::microsec_clock::local_time();
    int num_edges = (int)rag->get_num_edges();
    vector<Rag_t::edges_iterator> iter_vec;

    for (Rag_t::edges_iterator iter = rag->edges_begin(); iter != rag->edges_end(); ++iter) {
        iter_vec.push_back(iter);
    }


    cilk_for (int i = 0; i < num_edges; i++) {
        Rag_t::edges_iterator iter = iter_vec[i];
        if ( (!(*iter)->is_preserve()) && (!(*iter)->is_false_edge()) ) {
            double prev_val = (*iter)->get_weight();    
            double val = feature_mgr->get_prob(*iter);
            (*iter)->set_weight(val);

            // (*iter)->set_property("qloc", edgeCount);

            // Node_t node1 = (*iter)->get_node1()->get_node_id();  
            // Node_t node2 = (*iter)->get_node2()->get_node_id();  
        }
    }

    int i = 0;
    for (Rag_t::edges_iterator iter = rag->edges_begin(); iter != rag->edges_end(); ++iter) {
        if ( (!(*iter)->is_preserve()) && (!(*iter)->is_false_edge()) ) {
            (*iter)->set_property("qloc", i);
            i++;
        }
    }

    /*
    for (Rag_t::edges_iterator iter = rag->edges_begin(); iter != rag->edges_end(); ++iter) {
        if ( (!(*iter)->is_preserve()) && (!(*iter)->is_false_edge()) ) {
    	    double prev_val = (*iter)->get_weight();	
                double val = feature_mgr->get_prob(*iter);
                (*iter)->set_weight(val);

                (*iter)->set_property("qloc", edgeCount);

    	    // Node_t node1 = (*iter)->get_node1()->get_node_id();	
    	    // Node_t node2 = (*iter)->get_node2()->get_node_id();	

    	    edgeCount++;
    	}
    }
    */
    now = boost::posix_time::microsec_clock::local_time();
    cout << endl << "------------------------ AGGLO MID LOOP: " << (now - start).total_milliseconds() << " ms\n";

    start = boost::posix_time::microsec_clock::local_time();
    agglomerate_stack_parallel(stack, threshold, use_mito, true);
    now = boost::posix_time::microsec_clock::local_time();
    cout << endl << "------------------------ AGGLO SECOND PASS: " << (now - start).total_milliseconds() << " ms\n";
}

void agglomerate_stack_queue(Stack& stack, double threshold, 
                                bool use_mito, bool use_edge_weight)
{
    if (threshold == 0.0) {
        return;
    }

    RagPtr rag = stack.get_rag();
    FeatureMgrPtr feature_mgr = stack.get_feature_manager();

    vector<QE> all_edges;	    	
    int count=0; 	
    for (Rag_t::edges_iterator iter = rag->edges_begin(); iter != rag->edges_end(); ++iter) {
        if ( (!(*iter)->is_preserve()) && (!(*iter)->is_false_edge()) ) {


            RagNode_t* rag_node1 = (*iter)->get_node1();
            RagNode_t* rag_node2 = (*iter)->get_node2();

            Node_t node1 = rag_node1->get_node_id(); 
            Node_t node2 = rag_node2->get_node_id(); 

            double val;
            if(use_edge_weight)
                val = (*iter)->get_weight();
            else	
                val = feature_mgr->get_prob(*iter);    

            (*iter)->set_weight(val);
            (*iter)->set_property("qloc", count);

            QE tmpelem(val, make_pair(node1,node2));	
            all_edges.push_back(tmpelem); 

            count++;
        }
    }

    double error=0;  	

    MergePriorityQueue<QE> *Q = new MergePriorityQueue<QE>(rag.get());
    Q->set_storage(&all_edges);	

    PriorityQCombine node_combine_alg(feature_mgr.get(), rag.get(), Q); 

    while (!Q->is_empty()){
        QE tmpqe = Q->heap_extract_min();	

        //RagEdge_t* rag_edge = tmpqe.get_val();
        Node_t node1 = tmpqe.get_val().first;
        Node_t node2 = tmpqe.get_val().second;
        RagEdge_t* rag_edge = rag->find_rag_edge(node1,node2);;

        if (!rag_edge || !tmpqe.valid()) {
            continue;
        }
        double prob = tmpqe.get_key();
        if (prob>threshold)
            break;	

        RagNode_t* rag_node1 = rag_edge->get_node1();
        RagNode_t* rag_node2 = rag_edge->get_node2();
        node1 = rag_node1->get_node_id(); 
        node2 = rag_node2->get_node_id(); 

        if (use_mito) {
            if (is_mito(rag_node1) || is_mito(rag_node2)) {
                continue;
            }
        }

        // retain node1 
        stack.merge_labels(node2, node1, &node_combine_alg);
    }		



}

void agglomerate_stack_flat(Stack& stack, double threshold, bool use_mito)
{
    if (threshold == 0.0) {
        return;
    }

    RagPtr rag = stack.get_rag();
    FeatureMgrPtr feature_mgr = stack.get_feature_manager();

    vector<QE> priority;	    	
    FlatCombine node_combine_alg(feature_mgr.get(), rag.get(), &priority); 
    
    for(int ii=0; ii< priority.size(); ++ii) {
	QE tmpqe = priority[ii];	
        Node_t node1 = tmpqe.get_val().first;
        Node_t node2 = tmpqe.get_val().second;
	if(node1==node2)
	    continue;
	
        RagEdge_t* rag_edge = rag->find_rag_edge(node1,node2);;

        if (!rag_edge || !(priority[ii].valid()) || (rag_edge->get_weight())>threshold ) {
            continue;
        }

        RagNode_t* rag_node1 = rag_edge->get_node1();
        RagNode_t* rag_node2 = rag_edge->get_node2();
        if (use_mito) {
            if (is_mito(rag_node1) || is_mito(rag_node2)) {
                continue;
            }
        }
        
        node1 = rag_node1->get_node_id(); 
        node2 = rag_node2->get_node_id(); 
        
        stack.merge_labels(node2, node1, &node_combine_alg);
    }
}

void agglomerate_stack_mito(Stack& stack, double threshold)
{
    double error=0;  	

    RagPtr rag = stack.get_rag();
    FeatureMgrPtr feature_mgr = stack.get_feature_manager();

    MergePriority* priority = new MitoPriority(feature_mgr.get(), rag.get());
    priority->initialize_priority(threshold);
    
    DelayedPriorityCombine node_combine_alg(feature_mgr.get(), rag.get(), priority); 

    while (!(priority->empty())) {
        RagEdge_t* rag_edge = priority->get_top_edge();

        if (!rag_edge) {
            continue;
        }

        RagNode_t* rag_node1 = rag_edge->get_node1();
        RagNode_t* rag_node2 = rag_edge->get_node2();

        MitoTypeProperty mtype1, mtype2;
	try {    
            mtype1 = rag_node1->get_property<MitoTypeProperty>("mito-type");
        } catch (ErrMsg& msg) {
        
        }
        try {    
            mtype2 = rag_node2->get_property<MitoTypeProperty>("mito-type");
        } catch (ErrMsg& msg) {
        
        }
        if ((mtype1.get_node_type()==2) && (mtype2.get_node_type()==1))	{
            RagNode_t* tmp = rag_node1;
            rag_node1 = rag_node2;
            rag_node2 = tmp;		
        } else if ((mtype2.get_node_type()==2) && (mtype1.get_node_type()==1))	{
            // nophing	
        } else {
            continue;
        }

        Node_t node1 = rag_node1->get_node_id(); 
        Node_t node2 = rag_node2->get_node_id();

        // retain node1 
        stack.merge_labels(node2, node1, &node_combine_alg);
    }

    delete priority;
}

}
