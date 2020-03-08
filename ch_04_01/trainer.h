#ifndef TRAINER_INCLUDE 
#define TRAINER_INCLUDE

#include "gan.h"
#include <torch/torch.h>

// http://aidiary.hatenablog.com/entry/20180304/1520172429: detachの意味が書いてある。

template<typename Loader>
class Trainer
{
public:
    Trainer(
        Loader& loader, 
        GAN& gan, 
        int epochs, 
        const torch::Device& device, 
        int log_interval, 
        int save_interval, 
        int batches_per_epoch) 
        : loader_{loader}
        , gan_{gan}
        , epochs_{epochs}
        , device_{device}
        , log_interval_{log_interval}
        , save_interval_{save_interval}
        , batches_per_epoch_{batches_per_epoch} 
        , batch_size_{static_cast<int64_t>(loader_->options().batch_size)}
        , real_labels_{torch::ones({batch_size_, 1}).to(device_)}
        , fake_labels_{torch::zeros({batch_size_, 1}).to(device_)}
        , optimizer_for_discriminator_{
            select_optimizer(
                "rmsprop", 
                gan->get_discriminator()->parameters(), 
                gan->get_discriminator_params().learning_rate_)}
        , optimizer_for_generator_{
            select_optimizer(
                "rmsprop", 
                gan->get_generator()->parameters(), 
                gan->get_generator_params().learning_rate_)}
    {}

    void train()
    {
        gan_->to(device_);
        gan_->train();
        //gan->get_generator()->train();
        //gan->get_discriminator()->train();
       
        // train
        for (auto epoch = 1; epoch <= epochs_; ++epoch)
        {
            train(epoch);
        }
    }

private:
    Loader&                 loader_;
    GAN&                    gan_; 
    int                     epochs_; 
    const torch::Device&    device_; 
    int                     log_interval_; 
    int                     save_interval_;
    int                     batches_per_epoch_;
    int64_t                 batch_size_;
    torch::Tensor           real_labels_;
    torch::Tensor           fake_labels_;
    std::unique_ptr<torch::optim::Optimizer> optimizer_for_discriminator_;
    std::unique_ptr<torch::optim::Optimizer> optimizer_for_generator_;
    
    void train(int epoch)
    {
        float d_real_loss {};
        float d_fake_loss {};
        float g_loss {};
        int batch_index = 0;
        for (const auto& batch : *loader_)
        {
            const auto data = batch.data.to(device_);
            std::tie(d_real_loss, d_fake_loss) = train_discriminator(data);
            g_loss = train_generator();
            
            if (batch_index % log_interval_ == 0)
            {
                std::printf(
                    "\r[%2d/%2d][%4d/%4d] D_real_loss: %.4f | D_fake_loss: %.4f | D_loss: %.4f | G_loss: %.4f\n",
                    epoch,
                    epochs_,
                    batch_index,
                    batches_per_epoch_,
                    d_real_loss,
                    d_fake_loss,
                    d_real_loss + d_fake_loss,
                    g_loss);
            }

            if (batch_index % save_interval_ == 0)
            {
                // code to save
            }
            ++batch_index;
        }
    }
    
    std::tuple<float, float>  train_discriminator(const torch::Tensor& real_image)
    {
        optimizer_for_discriminator_->zero_grad();
        
        // consider real image
        const auto real_output = gan_->get_discriminator()->forward(real_image);
        const auto real_labels = torch::ones({real_output.size(0), 1}).to(device_);
        /* real_labels_ cannot be used, because batch size may be changed. */
        const auto real_loss = torch::binary_cross_entropy(real_output, real_labels); 
        real_loss.backward();

        // consider generated image
        const auto noise = torch::randn({batch_size_, gan_->get_z_dim()}, device_);
        const auto fake_image = gan_->get_generator()->forward(noise);
        const auto fake_output = gan_->get_discriminator()->forward(fake_image.detach());
        const auto fake_loss = torch::binary_cross_entropy(fake_output, fake_labels_); 
        fake_loss.backward();

        // update discriminator's parameters
        optimizer_for_discriminator_->step();

        return std::make_tuple(real_loss.template item<float>(), fake_loss.template item<float>());
    }

    float train_generator()
    {
        optimizer_for_generator_->zero_grad();
        const auto noise = torch::randn({batch_size_, gan_->get_z_dim()}, device_);
        const auto fake_image = gan_->get_generator()->forward(noise);
        const auto fake_output = gan_->get_discriminator()->forward(fake_image);
        const auto loss = torch::binary_cross_entropy(fake_output, real_labels_); 
        loss.backward();
        
        // update generator's parameters
        optimizer_for_generator_->step();

        return loss.template item<float>();
    }
    
    std::unique_ptr<torch::optim::Optimizer> select_optimizer(
        const std::string& name, 
        const std::vector<torch::Tensor>& params, 
        double learning_rate)
    {
        if (name == "adam")
        {
            return std::make_unique<torch::optim::Adam>(params, torch::optim::AdamOptions(learning_rate).beta1(0.5));
        }
        else if (name == "rmsprop")
        {
            return std::make_unique<torch::optim::RMSprop>(params, torch::optim::RMSpropOptions(learning_rate));
        }
        else
        {
            return std::make_unique<torch::optim::Adam>(params, torch::optim::AdamOptions(learning_rate));
        }
    }
};


#endif // TRAINER_INCLUDE 
