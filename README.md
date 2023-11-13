# GradAnalysis for Bfloat16 and Float16
Gradient Analysis for Neural Networks

## Summuary
Gradient Analysis is very important to find any gradient instabilty during training process of neural networks. If the gradients is becoming vanishing, the neural networks training will be ruined. 


# Gradient Analysis Sources

    class GradientVisualization(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            with tf.GradientTape() as tape:
                logits = model(train_images, training=True)
                loss_value = tf.keras.losses.sparse_categorical_crossentropy(train_labels, logits, from_logits=True)
            grads = tape.gradient(loss_value, model.trainable_weights)
    
            flattened_gradients = []
            for g in grads:
                flattened_gradients.append(tf.reshape(g, [-1]).numpy())
            flattened_gradients = np.concatenate(flattened_gradients, axis=0)
    
            # Count the number of zeros in the flattened_gradients
            # Count the number of zeros in the flattened_gradients
            print()
            print()
            zero_count = np.sum(flattened_gradients == 0)
            print("The number of 0:", zero_count)
            
            flattened_gradients = tf.cast(flattened_gradients, tf.float32)
            # Compute the standard deviation of flattened_gradients using numpy
            standard_deviation = np.std(flattened_gradients, ddof=1)
            print("Standard Deviation:", standard_deviation)
    
            # Compute the variance of flattened_gradients using numpy
            variance = np.var(flattened_gradients, ddof=1)
            print("Variance:", variance)
    
            # Compute the maximum and minimum values of flattened_gradients using numpy
            max_value = np.max(flattened_gradients)
            min_value = np.min(flattened_gradients)
            print("Max Value:", max_value)
            print("Min Value:", min_value)
            print()
            print()
            global arr1, arr2, arr3, arr4, arr5
            arr1.append(zero_count)
            arr2.append(standard_deviation)
            arr3.append(variance)
            arr4.append(max_value)
            arr5.append(min_value)
    
            #self.visualize_gradients(epoch, flattened_gradients)
    
        def visualize_gradients(self, epoch, gradients):
            plt.figure(figsize=(8, 6))
            plt.hist(gradients, bins=100, color='blue', alpha=0.7)
            plt.title(f"Gradient Distribution for Epoch: {epoch}")
            plt.xlabel('Gradient Magnitude')
            plt.ylabel('Count')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.tight_layout()
            plt.show()
