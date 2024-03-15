import tensorflow as tf
"""
   Regular_grid_2d: Generates a regular 2D grid of points within the range [-1, 1] 
   in both dimensions, with a specified height and width.

   grid_transform: Applies a transformation represented by a set of parameters (theta) 
   to a grid of points. 

   grid_sample_2d: Samples values from a moving image based on a grid of points. 
   It performs bilinear interpolation to get values at non-integer grid locations.
"""

def regular_grid_2d(height, width):
    # Generate linearly spaced values in the range [-1, 1] for x and y coordinates
    x = tf.linspace(-1.0, 1.0, width)  # shape (W, )
    y = tf.linspace(-1.0, 1.0, height)  # shape (H, )

    # Create a grid of points by combining x and y coordinates
    X, Y = tf.meshgrid(x, y)  # shape (H, W), both X and Y

    # Stack X and Y to form a grid tensor
    grid = tf.stack([X, Y], axis=-1)
    return grid

def grid_transform(theta, grid):
    # Get the number of batches, height, and width of the grid
    nb = tf.shape(theta)[0]
    nh, nw, _ = tf.shape(grid)
    
    # Extract x and y coordinates from the grid
    x = grid[..., 0]  # h,w
    y = grid[..., 1]
    
    # Flatten the x and y coordinates
    x_flat = tf.reshape(x, shape=[-1]) 
    y_flat = tf.reshape(y, shape=[-1])
    
    # Create a homogeneous representation of the grid coordinates
    ones = tf.ones_like(x_flat)
    grid_flat = tf.stack([x_flat, y_flat, ones])
    grid_flat = tf.expand_dims(grid_flat, axis=0)
    grid_flat = tf.tile(grid_flat, tf.stack([nb, 1, 1]))

    # Cast theta and grid_flat to float32
    theta = tf.cast(theta, 'float32')
    grid_flat = tf.cast(grid_flat, 'float32')

    # Apply transformation theta to grid_flat
    grid_new = tf.matmul(theta, grid_flat)  # n, 2, h*w
    
    # Reshape and transpose grid_new to match the original shape
    grid_new = tf.transpose(grid_new, perm=[0,2,1])
    grid_new = tf.reshape(grid_new, [nb, nh, nw, 2])

    return grid_new

def grid_sample_2d(moving, grid):
    # Get the dimensions of the moving image
    nb, nh, nw, nc = tf.shape(moving)

    # Extract x and y coordinates from the grid
    x = grid[..., 0]  # shape (N, H, W)
    y = grid[..., 1]
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')

    # Scale x and y from [-1.0, 1.0] to [0, W] and [0, H] respectively.
    x = (x + 1.0) * 0.5 * tf.cast(nw-1, 'float32')
    y = (y + 1.0) * 0.5 * tf.cast(nh-1, 'float32')

    # Convert coordinates to integers for indexing
    y_max = tf.cast(nh - 1, 'int32')
    x_max = tf.cast(nw - 1, 'int32')
    zero = tf.constant(0, 'int32')

    # Calculate the indices of the four nearest corners
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # Ensure indices are within the image boundaries
    x0 = tf.clip_by_value(x0, zero, x_max)
    x1 = tf.clip_by_value(x1, zero, x_max)
    y0 = tf.clip_by_value(y0, zero, y_max)
    y1 = tf.clip_by_value(y1, zero, y_max)

    # Collect indices of the four corners
    b = tf.ones_like(x0) * tf.reshape(tf.range(nb), [nb, 1, 1])
    idx_a = tf.stack([b, y0, x0], axis=-1)  # all top-left corners
    idx_b = tf.stack([b, y1, x0], axis=-1)  # all bottom-left corners
    idx_c = tf.stack([b, y0, x1], axis=-1)  # all top-right corners
    idx_d = tf.stack([b, y1, x1], axis=-1)  # all bottom-right corners

    # Gather values at the corners
    moving_a = tf.gather_nd(moving, idx_a)  # all top-left values
    moving_b = tf.gather_nd(moving, idx_b)  # all bottom-left values
    moving_c = tf.gather_nd(moving, idx_c)  # all top-right values
    moving_d = tf.gather_nd(moving, idx_d)  # all bottom-right values

    # Calculate weights for bilinear interpolation
    x0_f = tf.cast(x0, 'float32')
    x1_f = tf.cast(x1, 'float32')
    y0_f = tf.cast(y0, 'float32')
    y1_f = tf.cast(y1, 'float32')
    
    wa = tf.expand_dims((x1_f - x) * (y1_f - y), axis=-1)
    wb = tf.expand_dims((x1_f - x) * (y - y0_f), axis=-1)
    wc = tf.expand_dims((x - x0_f) * (y1_f - y), axis=-1)
    wd = tf.expand_dims((x - x0_f) * (y - y0_f), axis=-1)
    
    # Perform bilinear interpolation
    moved = tf.add_n([wa * moving_a, wb * moving_b, wc * moving_c, wd * moving_d])
    
    return moved
