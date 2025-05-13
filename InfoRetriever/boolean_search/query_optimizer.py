"""
Query optimization module for boolean search
"""

from .parser import TermNode, AndNode, OrNode, NotNode

def estimate_size(node, index):
    """
    Estimate the result size of a node
    
    Args:
        node: AST node to evaluate
        index: Index dictionary
        
    Returns:
        Estimated size of the result
    """
    if isinstance(node, TermNode):
        # For terms, return the actual document count
        return len(index.get(node.value, set()))
    
    elif isinstance(node, NotNode):
        # NOT operations are expensive - high estimate
        child_size = estimate_size(node.child, index)
        return max(0, len(index.keys()) - child_size)
    
    elif isinstance(node, AndNode):
        # For AND, estimate as the minimum of the sizes
        left_size = estimate_size(node.left, index)
        right_size = estimate_size(node.right, index)
        return min(left_size, right_size)
    
    elif isinstance(node, OrNode):
        # For OR, estimate as the sum of sizes (capped by total docs)
        left_size = estimate_size(node.left, index)
        right_size = estimate_size(node.right, index)
        all_docs_size = len(set().union(*index.values()))
        return min(left_size + right_size, all_docs_size)
    
    # Default case
    return float('inf')

def reorder_query_ast(node, index):
    """
    Reorder the query AST to optimize execution by evaluating smaller sets first
    
    Args:
        node: Root node of the AST
        index: Index dictionary
    
    Returns:
        Optimized AST
    """
    # Base cases
    if isinstance(node, TermNode) or isinstance(node, NotNode):
        return node
    
    # Recursive case: AND or OR nodes
    if isinstance(node, AndNode) or isinstance(node, OrNode):
        # First reorder children recursively
        reordered_left = reorder_query_ast(node.left, index)
        reordered_right = reorder_query_ast(node.right, index)
        
        # Estimate sizes for reordered children
        left_size = estimate_size(reordered_left, index)
        right_size = estimate_size(reordered_right, index)
        
        # For AND, put smaller operand first to enable short-circuiting
        if isinstance(node, AndNode) and right_size < left_size:
            return AndNode(reordered_right, reordered_left)
            
        # For OR, order doesn't matter as much, but we can still reorder for consistency
        if isinstance(node, OrNode) and right_size < left_size:
            return OrNode(reordered_right, reordered_left)
            
        # Use reordered children but keep original order
        if isinstance(node, AndNode):
            return AndNode(reordered_left, reordered_right)
        else:  # OrNode
            return OrNode(reordered_left, reordered_right)
    
    # Default case - return unchanged
    return node
