#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "lapack_wrapper.h"

#define K 2 // Number of Clusters
#define ITERATIONS 20 // Number of iterations of KMeans
#define NUM_ELEMENTS 2000 // Number of datapoints
#define D 2 // Dimension of data
#define EM_ITERATIONS 10 

double det (double * ftemp_var){
	double eigenval[D] = {0};
	double eigenvec[D*D] = {0};
	double temp = 1;
	eigen_decomposition(D, ftemp_var, eigenvec, eigenval);

	for(int dim=0; dim<D;dim++)
		temp = temp*eigenval[dim];

	return temp;
}

double phi(double *x, double *temp_mean, double *temp_var){
	double dot_prod,val=0; 	int status;
	double temp_x[D] ={0};
	for (int dim=0; dim<D; dim++)			
		temp_x[dim] = x[dim] - temp_mean[dim];
	double ftemp_var[D*D] = {0};
	double ftemp_var_inv[D*D] = {0};
	double vecB[D] = {0};
	cvec2fvec(D, D, temp_var, ftemp_var);
	status = matrix_invert(D,ftemp_var,ftemp_var_inv);
	matrix_vector_mult(D, D, 1, 0, ftemp_var_inv, temp_x, vecB );
	dot_prod=dotprod(D,temp_x,vecB);

	val = exp(-dot_prod/2)/pow(pow(2*M_PI,D)*det(ftemp_var),0.5);
}

double calculate_distance(double *d1, double *d2){
    double dist = 0;
	for (int i = 0; i < D; i++)
		dist += (d1[i] - d2[i])*(d1[i] - d2[i]);
	return dist;
}

double* element_k(double *data, int col){
	double* col_k = (double *)malloc(D*sizeof(double));
	for(int i=0; i<D; i++){
		col_k[i] = data[col*D + i];
	}
	return col_k;
}

int calculate_closest_centroid(double *data, double *means){
	double min_dist = RAND_MAX, dist_i;
	int closest_centroid;
	for(int i=0; i<K; i++){
		dist_i = calculate_distance(data, element_k(means,i));
		if(dist_i<min_dist){
			min_dist = dist_i;
			closest_centroid = i;
		}
	}
	return closest_centroid;
}

double* compute_centroids(double *data, int* cluster_label){
	double *means = (double *)malloc(K*D*sizeof(double));
	int *cluster_count = (int *)malloc(K*sizeof(int));
	for(int i=0; i<NUM_ELEMENTS; i++){
		cluster_count[cluster_label[i]]++;
		for(int j=0; j<D; j++){
			means[cluster_label[i]*D +j] += data[i*D +j];
		}
	}
	for(int i=0; i<K; i++){
		for(int j=0; j<D; j++){
			means[i*D +j] /= cluster_count[i];
		}
	}
	return means;
}

int calc_num_moved_pts(int *label_prev, int *label_ct){
	int count = 0;

	for(int i=0; i<NUM_ELEMENTS; i++){
		if(label_prev[i] != label_ct[i]){
			count++;
		}
	}
	return count;
}

int main(int argc, char** argv){
	int MYRANK,NO_OF_PROCS,ARRAY_LENGTH,rank,size,READ_START=0,root=0;

	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &MYRANK);
  	MPI_Comm_size(MPI_COMM_WORLD, &NO_OF_PROCS);
  	rank=MYRANK;
  	size=NO_OF_PROCS;

  	MPI_Datatype filetype;
    MPI_File fin;
    MPI_Status status;
	/***** MPI PARALLEL IO to read input Data *****/
    int gsizes[2], distribs, dargs, psizes;
    int ndims = 1;

    gsizes[0] = NUM_ELEMENTS; /* no. of rows in global array */
    gsizes[1] = D; /* no. of columns in global array*/

	distribs = MPI_DISTRIBUTE_BLOCK; /* Distribited block cyclic distribution along rows */
    dargs = MPI_DISTRIBUTE_DFLT_DARG; 
    psizes = size; /* no. of processes in process grid */
	
	int num_elements_per_proc = NUM_ELEMENTS / size;
	int x_size = num_elements_per_proc*D;
	double *x = (double *)malloc(num_elements_per_proc*D*sizeof(double));

	MPI_Type_create_darray(size, rank, ndims, gsizes, &distribs, &dargs, &psizes, MPI_ORDER_C, MPI_DOUBLE, &filetype);
	MPI_Type_commit(&filetype);

	MPI_File_open(MPI_COMM_WORLD, "file_2k", MPI_MODE_RDONLY, MPI_INFO_NULL, &fin);
    MPI_File_set_view(fin, 0, MPI_DOUBLE, filetype, "native", MPI_INFO_NULL);
    MPI_File_read_all(fin, x, x_size, MPI_DOUBLE, &status);
    MPI_File_close(&fin);

    MPI_Comm comm = MPI_COMM_WORLD;
	
	ARRAY_LENGTH = num_elements_per_proc;

	double *means = (double *)malloc(K*D*sizeof(double));
	if(rank == root){
		for(int i=0;i<K;i++){
			int rand_num = rand()%num_elements_per_proc;
			for(int j=0;j<D;j++){
				float rand_float = ((float)rand()/(float)(RAND_MAX)) * 5;
				means[i*D+j] = x[rand_num*D+j] + rand_float;
			}
		}
		printf("\nInitial Means \n");
		for(int i=0;i<K*D;i++){
			printf("%lf ",means[i]);
		}
	}
	int *cluster_label_ct = (int *)malloc(NUM_ELEMENTS*sizeof(int));
	int *cluster_label_prev = (int *)malloc(NUM_ELEMENTS*sizeof(int));
	int flag = 0;
	int moved_pts, moved_pts_all;
	int count_all[K]={0};

	for(int t=0; t<ITERATIONS; t++){
		MPI_Bcast(&flag, 1, MPI_INT, root, comm);
		if(flag==1)
			break;
		MPI_Bcast(means, K*D, MPI_DOUBLE, root, comm);

		// Assign datapoints to clusters
		for(int i=0; i<num_elements_per_proc; i++){
			//assign pt to the closest centroid
			cluster_label_ct[i] = calculate_closest_centroid(element_k(x,i), means);
		}
		// Recompute the means

		double new_means[K*D] = {0};
		double new_sum[K*D] = {0};
		int cluster_count[K] = {0};

		for(int i=0; i<num_elements_per_proc; i++){
			cluster_count[cluster_label_ct[i]]++;
			for(int j=0; j<D; j++){
				new_sum[cluster_label_ct[i]*D +j] += x[i*D +j];
			}
		}
		
		MPI_Reduce(new_sum, new_means, K*D, MPI_DOUBLE, MPI_SUM, root, comm);
		MPI_Reduce(cluster_count, count_all, K, MPI_DOUBLE, MPI_SUM, root, comm);

		if(rank == root){
			for(int i=0; i<K; i++){
				for(int j=0; j<D; j++){
					means[i*D +j] = new_means[i*D +j] / count_all[i];
				}
			}
		}

		if(t>0){
			moved_pts = calc_num_moved_pts(cluster_label_prev, cluster_label_ct);
		}
		for(int i=0; i<NUM_ELEMENTS; i++){
			cluster_label_prev[i] = cluster_label_ct[i];
		}
		MPI_Reduce(&moved_pts, &moved_pts_all, 1, MPI_INT, MPI_SUM, root, comm);
		if(rank==root && moved_pts_all==0)
			flag = 1;
	}
	
	// Initialize variance and Lambda
	double *loc_var = (double *)malloc(K*D*D*sizeof(double));
	double *temp_diff = (double *)malloc(D*sizeof(double));
	double temp;

	for(int i=0; i<num_elements_per_proc; i++){
		// cluster_count[cluster_label_ct[i]]++;
		for(int m=0; m<D; m++){
			temp_diff[m] = x[i*D +m] - means[cluster_label_ct[i]*D +m];
		}
		for(int j=0; j<D; j++){
			for(int jj=j; jj<D; jj++){
				temp = temp_diff[j]*temp_diff[jj];
				loc_var[cluster_label_ct[i]*D*D + j*D + jj] += temp;
				if(j!=jj)
					loc_var[cluster_label_ct[i]*D*D + jj*D + j] += temp;
			}
		}
	}
	double *var = (double *)malloc(K*D*D*sizeof(double));
	double *lam = (double *)malloc(K*sizeof(double));
	MPI_Reduce(loc_var, var, K*D*D, MPI_DOUBLE, MPI_SUM, root, comm);

	if(rank == root){
		for(int i=0; i<K; i++){
			lam[i] = (double)count_all[i]/NUM_ELEMENTS;
			printf("\n %lf", lam[i]);
			for(int j=0; j<D*D; j++){
				var[i*D*D + j] /= count_all[i];
			}
		}
	}
	MPI_Bcast(lam,K,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(means,D*K,MPI_DOUBLE,0,MPI_COMM_WORLD);		
	MPI_Bcast(var,D*D*K,MPI_DOUBLE,0,MPI_COMM_WORLD);
	

	if(rank==root){
		printf("\n Means \n");
		for(int i=0;i<K*D;i++){
			printf("%lf ",means[i]);
		}
		printf("\n Initial Covariance Matrix \n");
		for(int i=0; i<K; i++){
			printf(" Cluster %d\n", i);
			for(int j=0; j<D; j++){
				for(int k=0; k<D; k++){
					printf("%lf\t",var[i*D*D + j*D + k]);
				}
				printf("\n");
			}
		}
	}
	MPI_Barrier(comm);

	double *gamma = (double *)malloc(ARRAY_LENGTH*K*sizeof(double));
	double *temp_x = (double *)malloc(D*sizeof(double));
	double *temp_var = (double *)malloc(D*D*sizeof(double));
	double *temp_mean = (double *)malloc(D*sizeof(double));
		
		for (int i=0; i<ARRAY_LENGTH; i++){
		double den = 0;

		for (int dim=0; dim<D; dim++){
			temp_x[dim] = x[READ_START + i*D + dim];
		}
		
		for(int j=0; j<K; j++){
			for (int dim=0; dim<D; dim++){
				temp_mean[dim] = means[j*D+dim];
			}
			for (int dim=0; dim<D*D; dim++){
				temp_var[dim] = var[j*D*D+dim];
			}

			gamma[i*K+j] = lam[j]*phi(temp_x,temp_mean,temp_var);
			den = den + gamma[i*K+j];
		}
		for(int j=0; j<K; j++){
			gamma[i*K+j] = gamma[i*K+j]/den;
		}			
	} // all initial values have been computed

	for (int iter=0; iter<EM_ITERATIONS; iter++){
	 	double send_array[K*(D+D*D+1)];
	 	double recv_array[K*(D+D*D+1)];
				
	 	for (int j=0; j<K; j++){
	 		double num_mean[D] = {0};
	 		double num_var[D*D] = {0};
	 		double den = 0;
		//	printf("for cluster j=%d\n",j);
	 		for (int i=0; i<ARRAY_LENGTH; i++){		
	 			for(int dim=0; dim<D; dim++){
	 				num_mean[dim] = num_mean[dim] + gamma[i*K+j]*x[READ_START+i*D+dim];
	 			}

	 			if (D==1)
	 				num_var[0] = num_var[0] + gamma[i*K+j]*pow(x[READ_START+i*D]-means[j*D],2);
				else{
	 				for (int dimx=0; dimx<D; dimx++){
	 					for(int dimy=0; dimy<D; dimy++)
	 						num_var[dimx*D + dimy] += gamma[i*K+j]*(x[READ_START+i*D+dimx]-means[j*D + dimx])*(x[READ_START+i*D+dimy]-means[j*D+dimy]);
	 				}
				}
	 			den = den + gamma[i*K+j];
	 		}
	 		for (int dim=0; dim<D;dim++)
		 		send_array[j*(D+D*D+1)+dim] = num_mean[dim];
				
	 		for (int dim=0; dim<D*D; dim++)
	 			send_array[j*(D+D*D+1)+D+dim] = num_var[dim];
	
	 		send_array[j*(D+D*D+1)+D+D*D] = den;
	 	}

	 	MPI_Barrier(MPI_COMM_WORLD);
	 	MPI_Reduce( send_array  ,recv_array  , K*(D+D*D+1) , MPI_DOUBLE, MPI_SUM, 0 , MPI_COMM_WORLD );
		
	 	if (MYRANK == 0){
	 		for (int j=0; j<K; j++){
	 			double num_mean[D] = {0};
	 			double num_var[D*D] = {0};
	 			double den = 0;
	 			for(int dim=0; dim<D; dim++)
	 				num_mean[dim] = recv_array[j*(D+D*D+1)+dim];
				
	 			for (int dim=0; dim<D*D; dim++)
	 				num_var[dim] = recv_array[j*(D+D*D+1)+D+dim];

	 			den = recv_array[j*(D+D*D+1)+D+D*D];
				
	 			lam[j] = den/NUM_ELEMENTS;    //den
	 
	 			for (int dim=0; dim<D; dim++){
	 				means[j*D+dim] = num_mean[dim]/den;
	 			}		
		
	 			for (int dim=0; dim<D*D; dim++){
	 				var[j*D*D+dim] = num_var[dim]/den;
	 			}
	 		}
	 	}
		
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Bcast(lam,K,MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Bcast(means,D*K,MPI_DOUBLE,0,MPI_COMM_WORLD);		
		MPI_Bcast(var,D*D*K,MPI_DOUBLE,0,MPI_COMM_WORLD);		
		
		for (int i=0; i<ARRAY_LENGTH; i++){
			double den = 0;
			for (int dim=0; dim<D; dim++)
				temp_x[dim] = x[READ_START+i*D+dim];
			
			for(int j=0; j<K; j++){
				
				for (int dim=0; dim<D; dim++)
					temp_mean[dim] = means[j*D+dim];
		
				for (int dim=0; dim<D*D; dim++)
					temp_var[dim] = var[j*D*D+dim]; 

				gamma[i*K+j] = lam[j]*phi(temp_x,temp_mean,temp_var);
				den = den + gamma[i*K+j];
			}	
			for(int j=0; j<K; j++)
				gamma[i*K+j] = gamma[i*K+j]/den;		
		}
		MPI_Barrier(MPI_COMM_WORLD);
	} //end of iterations

	if (MYRANK==0){	
		for(int j=0; j<K; j++){
			printf("\n\n\n myrank=%d",MYRANK);

			printf("\nCLuster No = %d",j);
			
			printf("\nlambda= %f",lam[j]);	
			
			for (int dim=0; dim<D; dim++)
			printf("\nmean= %f",means[j*D+dim]);
			
			for (int dim=0; dim<D*D; dim++)
			printf("\nvar=%f",var[j*D*D+dim]);
			
			printf("\n");
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	return 0;	
}