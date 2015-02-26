#include "../FeatureManager/FeatureMgr.h"
#include "BioStack.h"
#include "MitoTypeProperty.h"

#include <json/value.h>
#include <json/reader.h>
#include <vector>

#include <cilk/cilk.h>

#include <time.h>
#include <boost/thread/mutex.hpp>
#include <pthread.h>

using std::tr1::unordered_set;
using std::tr1::unordered_map;
using std::vector;


namespace NeuroProof {

unordered_set<Label_t> global_labels_set;
boost::mutex global_labels_set_mu;
boost::mutex mu;
boost::mutex labelvol_mu;
boost::mutex problist_mu;
// pthread_mutex_t mu;
// pthread_mutex_t labelvol_mu;
// pthread_mutex_t problist_mu;


VolumeLabelPtr BioStack::create_syn_label_volume()
{
    if (!labelvol) {
        throw ErrMsg("No label volume defined for stack");
    }

    return create_syn_volume(labelvol);
}

VolumeLabelPtr BioStack::create_syn_gt_label_volume()
{
    if (!gt_labelvol) {
        throw ErrMsg("No GT label volume defined for stack");
    }

    return create_syn_volume(gt_labelvol);
}

VolumeLabelPtr BioStack::create_syn_volume(VolumeLabelPtr labelvol2)
{
    vector<Label_t> labels;

    for (int i = 0; i < synapse_locations.size(); ++i) {
        Label_t label = (*labelvol2)(synapse_locations[i][0],
            synapse_locations[i][1], synapse_locations[i][2]); 
        labels.push_back(label);
    }
    
    VolumeLabelPtr synvol = VolumeLabelData::create_volume();   
    synvol->reshape(VolumeLabelData::difference_type(labels.size(), 1, 1));

    for (int i = 0; i < labels.size(); ++i) {
        synvol->set(i, 0, 0, labels[i]);  
    }
    return synvol;
}

void BioStack::load_saved_synapse_counts(unordered_map<Label_t, int>& synapse_counts)
{
    saved_synapse_counts = synapse_counts;
}

void BioStack::load_synapse_counts(unordered_map<Label_t, int>& synapse_counts)
{
    for (int i = 0; i < synapse_locations.size(); ++i) {
        Label_t body_id = (*labelvol)(synapse_locations[i][0],
                synapse_locations[i][1], synapse_locations[i][2]);

        if (body_id) {
            synapse_counts[body_id]++;
        }
    }
}

void BioStack::load_synapse_labels(unordered_set<Label_t>& synapse_labels)
{
    for (int i = 0; i < synapse_locations.size(); ++i) {
        Label_t body_id = (*labelvol)(synapse_locations[i][0],
                synapse_locations[i][1], synapse_locations[i][2]);
        synapse_labels.insert(body_id);
    }
}

void BioStack::read_prob_list(std::string prob_filename, std::string dataset_name)
{
    prob_list = VolumeProb::create_volume_array(prob_filename.c_str(), dataset_name.c_str());
    cout << "Read prediction array" << endl;  
    
}

bool BioStack::is_mito(Label_t label)
{
    RagNode_t* rag_node = rag->find_rag_node(label);

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
void BioStack::set_classifier()
{
    assert(feature_manager);
    EdgeClassifier* eclfr = new OpencvRFclassifier();	
    feature_manager->set_classifier(eclfr);
}

void BioStack::save_classifier(std::string clfr_name)
{
    assert(feature_manager);
    feature_manager->get_classifier()->save_classifier(clfr_name.c_str());
}


void BioStack::build_rag_loop(unordered_map<Label_t, MitoTypeProperty> &mito_probs, 
                                            int x_start, int x_end, int y_start, int y_end, int z_start, int z_end)
{
    unordered_set<Label_t> labels;
    unsigned int maxx = get_xsize() - 1; 
    unsigned int maxy = get_ysize() - 1; 
    unsigned int maxz = get_zsize() - 1; 

    unordered_set<Label_t> my_labels_set;
    unordered_set<Label_t> their_labels_set;

    map<pair<Label_t, Label_t>, vector<vector<double> > > edge_to_pred_map;

    RagPtr my_temp_rag = RagPtr(new Rag_t());

    // cilk_for (int z = z_start; z < z_end; z++) {
    for (int z = z_start; z < z_end; z++) {    
        vector<double> predictions(prob_list.size(), 0.0);
        for (int y = y_start; y < y_end; y++)
            for (int x = x_start; x < x_end; x++) {
                // labelvol_mu.lock();
                Label_t label = (*labelvol)(x,y,z); 
                
                if (!label) {
                    // labelvol_mu.unlock();
                    continue;
                }



                /*
                if (my_labels_set.find(label) == my_labels_set.end()) {
                    // if we know this belongs to someone else
                    if (their_labels_set.find(label) != their_labels_set.end())
                        continue;

                    bool claimed = false;
                    global_labels_set_mu.lock();
                    if (global_labels_set.find(label) == global_labels_set.end()) {
                        global_labels_set.insert(label);
                    } else {
                        claimed = true;
                    }
                    global_labels_set_mu.unlock();

                    if (claimed) {
                        their_labels_set.insert(label);
                        // continue;
                    } else {
                        my_labels_set.insert(label);
                    }
                }
                */

                RagNode_t * temp_node = rag->find_rag_node(label);

                if (!temp_node)
                    temp_node = my_temp_rag->insert_rag_node(label);

                temp_node->incr_size();


                /*
                Label_t label2 = 0, label3 = 0, label4 = 0, label5 = 0, label6 = 0, label7 = 0;
                if (x > 0) label2 = (*labelvol)(x-1,y,z);
                if (x < maxx) label3 = (*labelvol)(x+1,y,z);
                if (y > 0) label4 = (*labelvol)(x,y-1,z);
                if (y < maxy) label5 = (*labelvol)(x,y+1,z);
                if (z > 0) label6 = (*labelvol)(x,y,z-1);
                if (z < maxz) label7 = (*labelvol)(x,y,z+1);
                */

                set<Label_t> neighbors;
                if (x > 0)
                    neighbors.insert((*labelvol)(x-1,y,z));
                if (x < maxx)
                    neighbors.insert((*labelvol)(x+1,y,z));
                if (y > 0)
                    neighbors.insert((*labelvol)(x,y-1,z)); 
                if (y < maxy)
                    neighbors.insert((*labelvol)(x,y+1,z));
                if (z > 0)
                    neighbors.insert((*labelvol)(x,y,z-1));
                if (z < maxz)
                    neighbors.insert((*labelvol)(x,y,z+1));

                // labelvol_mu.unlock();


                for (unsigned int i = 0; i < prob_list.size(); ++i) {
                    predictions[i] = (*(prob_list[i]))(x,y,z);
                }


                mu.lock();


                // boost::mutex::scoped_lock scoped_lock(mu);
                
                RagNode_t * node = rag->find_rag_node(label);

                if (!node) {
                    node =  rag->insert_rag_node(label); 
                }

                node->incr_size();
            
                if (feature_manager) {
                    feature_manager->add_val(predictions, node);
                }
                mito_probs[label].update(predictions); 

                /*
                if (label2 && (label != label2)) {
                    rag_add_edge(label, label2, predictions);
                    labels.insert(label2);
                }
                if (label3 && (label != label3) && (labels.find(label3) == labels.end())) {
                    rag_add_edge(label, label3, predictions);
                    labels.insert(label3);
                }
                if (label4 && (label != label4) && (labels.find(label4) == labels.end())) {
                    rag_add_edge(label, label4, predictions);
                    labels.insert(label4);
                }
                if (label5 && (label != label5) && (labels.find(label5) == labels.end())) {
                    rag_add_edge(label, label5, predictions);
                    labels.insert(label5);
                }
                if (label6 && (label != label6) && (labels.find(label6) == labels.end())) {
                    rag_add_edge(label, label6, predictions);
                    labels.insert(label6);
                }
                if (label7 && (label != label7) && (labels.find(label7) == labels.end())) {
                    rag_add_edge(label, label7, predictions);
                }

                if (!label2 || !label3 || !label4 || !label5 || !label6 || !label7) {
                    node->incr_boundary_size();
                }
                labels.clear();    
                */

                for (set<Label_t>::iterator it = neighbors.begin(); it != neighbors.end(); ++it) {
                    rag_add_edge(label, *it, predictions);
                } 

                mu.unlock();            
            }
    }
    // cout << "LABELS: ";
    // for (unordered_set<Label_t>::iterator it = my_labels_set.begin(); it != my_labels_set.end(); ++it) {
    //     cout << ", " << *it;
    // }

    my_temp_rag.reset();
}


void test_cilk(vector<int> &data, int num) {
    data.push_back(num);
}


void BioStack::build_rag()
{
    if (get_prob_list().size()==0){
        Stack::build_rag();
        return;
    }

    if (!feature_manager){
        FeatureMgrPtr feature_manager_(new FeatureMgr(prob_list.size()));
        set_feature_manager(feature_manager_);
        feature_manager->set_basic_features(); 
    }
    
    //printf("Building bioStack rag\n");
    if (!labelvol) {
        throw ErrMsg("No label volume defined for stack");
    }

    rag = RagPtr(new Rag_t);

    vector<double> predictions(prob_list.size(), 0.0);
    unordered_set<Label_t> labels;
   
    unsigned int maxx = get_xsize() - 1; 
    unsigned int maxy = get_ysize() - 1; 
    unsigned int maxz = get_zsize() - 1; 
    unordered_map<Label_t, MitoTypeProperty> mito_probs;
 
    int x_full = (int)(*labelvol).shape(0);
    int y_full = (int)(*labelvol).shape(1);
    int z_full = (int)(*labelvol).shape(2);

    // pthread_mutex_init(&mu,NULL);
    // pthread_mutex_init(&problist_mu,NULL);
    // pthread_mutex_init(&labelvol_mu,NULL);

    int z_half = z_full/2;
    int z_fourth = z_full/4;
    // cilk_spawn build_rag_loop(mito_probs, 0, x_full, 0, y_full, 0, z_fourth);
    // cilk_spawn build_rag_loop(mito_probs, 0, x_full, 0, y_full, z_fourth, z_half);
    // cilk_spawn build_rag_loop(mito_probs, 0, x_full, 0, y_full, z_half, z_half + z_fourth);
    // build_rag_loop(mito_probs, 0, x_full, 0, y_full, z_half + z_fourth, z_full);

    cilk_spawn build_rag_loop(mito_probs, 0, x_full, 0, y_full, 0, z_half);
    build_rag_loop(mito_probs, 0, x_full, 0, y_full, z_half, z_full);

    cilk_sync;

    // build_rag_loop(mito_probs, 0, x_full, 0, y_full, 0, z_full);

    // vector<int> data;
    // cilk_spawn test_cilk(data, 5);
    // test_cilk(data, 10);

    // cilk_sync;

    // for (vector<int>::iterator it = data.begin(); it != data.end(); ++it)
    //     cout << "DATA: " << *it << endl;


    /*
    volume_forXYZ(*labelvol, x, y, z) {
        Label_t label = (*labelvol)(x,y,z); 
        
        if (!label) {
            continue;
        }

        RagNode_t * node = rag->find_rag_node(label);

        if (!node) {
            node =  rag->insert_rag_node(label); 
        }
        node->incr_size();
                
        for (unsigned int i = 0; i < prob_list.size(); ++i) {
            predictions[i] = (*(prob_list[i]))(x,y,z);
        }
        if (feature_manager) {
            feature_manager->add_val(predictions, node);
        }
        mito_probs[label].update(predictions); 

        Label_t label2 = 0, label3 = 0, label4 = 0, label5 = 0, label6 = 0, label7 = 0;
        if (x > 0) label2 = (*labelvol)(x-1,y,z);
        if (x < maxx) label3 = (*labelvol)(x+1,y,z);
        if (y > 0) label4 = (*labelvol)(x,y-1,z);
        if (y < maxy) label5 = (*labelvol)(x,y+1,z);
        if (z > 0) label6 = (*labelvol)(x,y,z-1);
        if (z < maxz) label7 = (*labelvol)(x,y,z+1);

        if (label2 && (label != label2)) {
            rag_add_edge(label, label2, predictions);
            labels.insert(label2);
        }
        if (label3 && (label != label3) && (labels.find(label3) == labels.end())) {
            rag_add_edge(label, label3, predictions);
            labels.insert(label3);
        }
        if (label4 && (label != label4) && (labels.find(label4) == labels.end())) {
            rag_add_edge(label, label4, predictions);
            labels.insert(label4);
        }
        if (label5 && (label != label5) && (labels.find(label5) == labels.end())) {
            rag_add_edge(label, label5, predictions);
            labels.insert(label5);
        }
        if (label6 && (label != label6) && (labels.find(label6) == labels.end())) {
            rag_add_edge(label, label6, predictions);
            labels.insert(label6);
        }
        if (label7 && (label != label7) && (labels.find(label7) == labels.end())) {
            rag_add_edge(label, label7, predictions);
        }

        if (!label2 || !label3 || !label4 || !label5 || !label6 || !label7) {
            node->incr_boundary_size();
        }
        labels.clear();
    }
    */
    
    Label_t largest_id = 0;
    for (Rag_t::nodes_iterator iter = rag->nodes_begin(); iter != rag->nodes_end(); ++iter) {
        Label_t id = (*iter)->get_node_id();
        largest_id = (id>largest_id)? id : largest_id;
	
        MitoTypeProperty mtype = mito_probs[id];
        mtype.set_type(); 
        (*iter)->set_property("mito-type", mtype);
    }
    //printf("Done Biostack rag, largest: %u\n", largest_id);
}



void BioStack::set_edge_locations()
{
  
    EdgeCount best_edge_z;
    EdgeLoc best_edge_loc;
    determine_edge_locations(best_edge_z, best_edge_loc, false); //optimal_prob_edge_loc);
    
    // set edge properties for export 
    for (Rag_t::edges_iterator iter = rag->edges_begin(); iter != rag->edges_end(); ++iter) {
//         if (!((*iter)->is_false_edge())) {
//             if (feature_manager) {
//                 double val = feature_manager->get_prob((*iter));
//                 (*iter)->set_weight(val);
//             } 
//         }
        Label_t x = 0;
        Label_t y = 0;
        Label_t z = 0;
        
        if (best_edge_loc.find(*iter) != best_edge_loc.end()) {
            Location loc = best_edge_loc[*iter];
            x = boost::get<0>(loc);
            // assume y is bottom of image
            // (technically ignored by raveler so okay)
            y = boost::get<1>(loc); //height - boost::get<1>(loc) - 1;
            z = boost::get<2>(loc);
        }
        
        (*iter)->set_property("location", Location(x,y,z));
    }
  
}

void BioStack::set_synapse_exclusions(vector<vector<unsigned int> >& synapse_locations_) 
{
    synapse_locations = synapse_locations_;
}

void BioStack::set_synapse_exclusions(const char* synapse_json)
{
    unsigned int ysize = labelvol->shape(1);

    if (!rag) {
        throw ErrMsg("No RAG defined for stack");
    }

    synapse_locations.clear();

    Json::Reader json_reader;
    Json::Value json_reader_vals;
    
    ifstream fin(synapse_json);
    if (!fin) {
        throw ErrMsg("Error: input file: " + string(synapse_json) + " cannot be opened");
    }
    if (!json_reader.parse(fin, json_reader_vals)) {
        throw ErrMsg("Error: Json incorrectly formatted");
    }
    fin.close();
 
    Json::Value synapses = json_reader_vals["data"];

    for (int i = 0; i < synapses.size(); ++i) {
        vector<vector<unsigned int> > locations;
        Json::Value location = synapses[i]["T-bar"]["location"];
        if (!location.empty()) {
            vector<unsigned int> loc;
            loc.push_back(location[(unsigned int)(0)].asUInt());
            loc.push_back(ysize - location[(unsigned int)(1)].asUInt() - 1);
            loc.push_back(location[(unsigned int)(2)].asUInt());
            synapse_locations.push_back(loc);
            locations.push_back(loc);
        }
        Json::Value psds = synapses[i]["partners"];
        for (int i = 0; i < psds.size(); ++i) {
            Json::Value location = psds[i]["location"];
            if (!location.empty()) {
                vector<unsigned int> loc;
                loc.push_back(location[(unsigned int)(0)].asUInt());
                loc.push_back(ysize - location[(unsigned int)(1)].asUInt() - 1);
                loc.push_back(location[(unsigned int)(2)].asUInt());
                synapse_locations.push_back(loc);
                locations.push_back(loc);
            }
        }

        for (int iter1 = 0; iter1 < locations.size(); ++iter1) {
            for (int iter2 = (iter1 + 1); iter2 < locations.size(); ++iter2) {
                add_edge_constraint(rag, labelvol, locations[iter1][0], locations[iter1][1],
                    locations[iter1][2], locations[iter2][0], locations[iter2][1], locations[iter2][2]);           
            }
        }
    }

}
    
void BioStack::serialize_graph_info(Json::Value& json_writer)
{
    unordered_map<Label_t, int> synapse_counts;
    if (saved_synapse_counts.size() > 0) {
        synapse_counts = saved_synapse_counts;
    } else { 
        load_synapse_counts(synapse_counts);
    }

    int id = 0;
    for (unordered_map<Label_t, int>::iterator iter = synapse_counts.begin();
            iter != synapse_counts.end(); ++iter, ++id) {
        Json::Value synapse_pair;
        synapse_pair[(unsigned int)(0)] = iter->first;
        synapse_pair[(unsigned int)(1)] = iter->second;
        json_writer["synapse_bodies"][id] =  synapse_pair;
    }
}

void BioStack::add_edge_constraint(RagPtr rag, VolumeLabelPtr labelvol2, unsigned int x1,
        unsigned int y1, unsigned int z1, unsigned int x2, unsigned int y2, unsigned int z2)
{
    Label_t label1 = (*labelvol2)(x1,y1,z1);
    Label_t label2 = (*labelvol2)(x2,y2,z2);

    if (label1 && label2 && (label1 != label2)) {
        RagEdge_t* edge = rag->find_rag_edge(label1, label2);
        if (!edge) {
            RagNode_t* node1 = rag->find_rag_node(label1);
            RagNode_t* node2 = rag->find_rag_node(label2);
            edge = rag->insert_rag_edge(node1, node2);
            edge->set_weight(1.0);
            edge->set_false_edge(true);
        }
        edge->set_preserve(true);
    }
}

}
