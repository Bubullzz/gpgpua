#include "image.hh"
#include "pipeline.hh"
#include "dispatcher.hh"
#include "fix_cpu.cuh"
#include "fix_gpu_handmade.cuh"
#include "fix_gpu_industrial.cuh"
#include "radix_sort.cuh"

#include <vector>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <filesystem>
#include <numeric>
#include <rmm/device_uvector.hpp>

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{
    test_radix_sort();
    return;

    // Choose processing mode here
    // {CPU, GPU_Handmade, GPU_Industrial} <-- copy-paste one of these
    const ProcessingMode mode = ProcessingMode::GPU_Industrial;
    print_mode<mode>();
    
    // -- Pipeline initialization

    std::cout << "File loading..." << std::endl;

    // - Get file paths

    using recursive_directory_iterator = std::filesystem::recursive_directory_iterator;
    std::vector<std::string> filepaths;
    for (const auto& dir_entry : recursive_directory_iterator("/afs/cri.epita.fr/resources/teach/IRGPUA/images"))
        filepaths.emplace_back(dir_entry.path());

    // - Init pipeline object

    Pipeline pipeline(filepaths);

    // -- Main loop containing image retring from pipeline and fixing

    const int nb_images = pipeline.images.size();
    std::vector<Image> images(nb_images);

    // - One CPU thread is launched for each image

    std::cout << "Done, starting compute" << std::endl;

    #pragma omp parallel for
    for (int i = 0; i < nb_images; ++i)
    {
        // TODO : make it GPU compatible (aka faster)
        // You will need to copy images one by one on the GPU
        // You can store the images the way you want on the GPU
        // But you should treat the pipeline as a pipeline :
        // You *must not* copy all the images and only then do the computations
        // You must get the image from the pipeline as they arrive and launch computations right away
        // There are still ways to speeds this process of course
        images[i] = pipeline.get_image(i);
        fix_image<mode>(images[i]);
    }

    std::cout << "Done with compute, starting computing total" << std::endl;

    // -- All images are now fixed : compute stats (total then sort)

    // - First compute the total of each image

    // DONE : make it GPU compatible (aka faster)
    compute_total<mode>(images, nb_images);

    std::cout << "Done with total, starting sort" << std::endl;
    
    // Sorting the images
    using ToSort = Image::ToSort;
    std::vector<ToSort> to_sort(nb_images);
    std::generate(to_sort.begin(), to_sort.end(), [n = 0, images] () mutable
    {
        return images[n++].to_sort;
    });

    sort<mode>(to_sort, nb_images);

    
    std::cout << "Done with sort !" << std::endl;

    // TODO : Test here that you have the same results
    // You can compare visually and should compare image vectors values and "total" values
    // If you did the sorting, check that the ids are in the same order

    // Reference totals from CPU execution
    std::vector<long long> expected_totals = {
        27805567, 185010925, 342970490, 33055988, 390348481, 91297791, 10825197,
        118842538, 72434629, 191735142, 182802772, 78632198, 491605096, 8109782,
        111786760, 406461934, 80671811, 70004942, 104275727, 30603818, 6496225,
        207334021, 268424419, 432916359, 51973720, 24489209, 80124196, 29256842,
        25803206, 34550754
    };
    std::cout << "Checking totals..." << std::endl;
    // Compare computed totals to expected ones
    for (int i = 0; i < nb_images; ++i)
    {
        long computed = images[i].to_sort.total;
        long expected = expected_totals[i];

        float diff = computed - expected;
        std::cout << "Image #" << images[i].to_sort.id
                << " total: " << computed
                << " (diff: " << diff << ")";

        if (diff == 0) {
            std::cout << " ✅" << std::endl;
        } else {
            std::cout << " ❌" << std::endl;
        }

        // You can keep the image writing if you still want the files
        std::ostringstream oss;
        oss << "Image#" << images[i].to_sort.id << ".pgm";
        images[i].write(oss.str());
    }

    // Check that images are sorted correctly
    std::cout << "Checking sorting..." << std::endl;
    for (int i = 0; i < nb_images - 1; ++i)
        if (to_sort[i].total > to_sort[i + 1].total)
        {
            std::cerr << "Sorting error on image id :" << to_sort[i].id << std::endl;
            return 1;
        }
    std::cout << "Sorting OK!  ✅" << std::endl;
    std::cout << "Done, the internet is safe now :)" << std::endl;

    // Cleaning
    // TODO : Don't forget to update this if you change allocation style
    for (int i = 0; i < nb_images; ++i)
    {
        cudaFreeHost(images[i].buffer);
        free(images[i].char_buffer);
    }
    return 0;
}
