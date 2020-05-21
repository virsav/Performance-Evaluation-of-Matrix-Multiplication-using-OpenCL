*******************************************************************************************************************************************************************************************************

CSE 520: Computer Architecture 2
Assignment 3

*******************************************************************************************************************************************************************************************************

ASU ID: 1217678787
NAME: Viraj Savaliya

*******************************************************************************************************************************************************************************************************

The Submission includes:

1. main.cpp = This is the modified main.cpp file to run your required configuration at a time.
	      It takes following 4 command line arguments in order:
	      	a.MM_CPU - 	1=CPU and 2=GPU
		b.MM_KERNEL - 	1=Naive and 2=Tiled
		c.SIZE - 	Size of the matrix (512, 1024, 2048)
		d.TILE_SIZE - 	Size of the tile (8,16)

2. matrix_mul.cl = This is the given OpenCL file which defines 2 kernels, namely Naive and Tiled.

3. Makefile = This is the file to make executable binary from the main.cpp file.

4. mm_all.sh = This is a bash script which includes all the commands to run the main.cpp file for all the required configurations of Assignment 3.

5. Report = The report includes the Obersvation table of execution time of all the required configurations
	    The CPU and GPU model information.
	    Console runtime screenshots showing the runtime command and the output message displayed.

*******************************************************************************************************************************************************************************************************

Instructions to use the program:

NOTE: Initially please transfer the given files into the directory you want to work from or open the terminal from within the directory where all the given files are in.

Please follow the below instructions in the same order =

	a. "make" - This command will create an executable binary for the program.
	b. "bash mm_all.sh" - This command will run the binary for all the required configurations of assignment 3.
	c. "make clean" - This command will delete the binary file created for the program.

NOTE: If you want to run the given program for your specific configuration then please use the following command:
	"./matrix_mul [MM_CPU] [MM_KERNEL] [SIZE] [TILE_SIZE]"

      The command will run the Reference C matrix multiplication and matrix multiplication for your given configuration.

      For MM_CPU other than 1/2 it will return Device not found.
      For MM_KERNEL other than 1/2 it will return Kernel not found.

*******************************************************************************************************************************************************************************************************