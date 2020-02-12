#include <bits/stdc++.h>
using namespace std;

struct Node {
	Node *left_node, *right_node;
	int left_index, right_index;
	int value;
}

Node *root;

int Query(Node *node, int left_query, int right_query) {
	if (left_query <= node->left_index && node->right_index <= right_query) {
		return node->value;
	}
	if (right_query <= node->left_index || node->right_index <= left_query) {
		return 0;
	}
	return Query(node->left_node, left_query, right_query) + Query(node->right_node, 
		left_query, right_query);
}

void Update(Node *node, int update_index, int new_value) {
	if (update_index < node->left_index || node->right_index <= update_index) {
		return;
	}
	if (node->right_index - node->left_index == 1) {
		node->value = new_value;
		return;
	}
	Update(node->left_node, update_index, new_value);
	Update(node->right_node, update_index, new_value);
	node->value = node->left_node->value + node->right_node->value;
}

vector<int> values;
void Build(Node *node) {
	if (node->right_index - node->left_index == 1) {
		node->value = values[node->left_index];
		return;
	}
	Node *left_child = node->left_node = new Node;
	Node *right_child = node->right_node = new Node;
	left_child->left_index = node->left_index;
	left_child->right_index = right_child->left_index = (node->left_index + node->right_index) / 2;
	right_child->right_index = node->right_index;
	Build(left_child);
	Build(right_child);
	node->value = node->left_child->value + node->right_child->value;
}

int main() {
	
}