#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//Le code du kernel
__global__ void gauss(double *a, int n, int currentItem)
{
    int j = blockIdx.x, k = threadIdx.x;
    if(j != currentItem){
       float pro = a[j * (n+1) + currentItem] / a[currentItem*(n+1) + currentItem];
       a[j*(n+1)+k] = a[j*(n+1)+k] - (a[currentItem*(n+1)+k]) * pro;
    }
}


int main()
{
    int n, i, j, k, flag = 0, c;
    double *a, akk, aik;
    double *cuda_a;
    //printf("Entrez la dimension du système:");
    scanf("%d",&n);
    int size = (n+1) * n * sizeof (double);
    a=(double *)malloc(size);
    /* saisie des coefficients de la matrice A */
    /*a[0] = 0.36; a[1] = 0.48; a[2] = 195; a[3] = 7; a[4] = 0.6;      a[5] = 9.92;
    a[6] = 0.29; a[7] = 0.46; a[8] = 105; a[9] = 3.68; a[10] = 1.66;      a[11] = 7.95;
    a[12] = 0.2; a[13] = 0.5; a[14] = 138; a[15] = 2.94; a[16] = 1.91;      a[17] = 7.54;
    a[18] = 0.32; a[19] = 0.46; a[20] = 10; a[21] = 2.63; a[22] = 0.81;      a[23] = 7.12;
    a[24] = 0.16; a[25] = 0.42; a[26] = 360; a[27] = 0.6; a[28] = 0.45;      a[29] = 7;*/
    
    printf("Entrez les coefficients de la matrice A, ligne par ligne :\n");
    for(i=0;i<n;i++) for(j=0;j<n;j++) {printf("Entrez le coefficient a(%d,%d)=",i+1,j+1); scanf("%lf",&a[i*(n+1)+j]);}
    printf("Rappel des coefficients saisis:\n");
    for(i=0;i<n;i++) {for(j=0;j<n;j++) printf("  a(%d,%d)=%.2lf",i+1,j+1,a[i*(n+1)+j]);printf("\n");}
    /* saisie des composantes du vecteur second membre b */
    printf("Entrez les composantes du vecteur second membre :\n");
    for(i=0;i<n;i++) {printf("Entrez la composante b(%d)=",i+1); scanf("%lf",&a[i*(n+1)+n]);}
    
    printf("Rappel des composantes saisies:\n");
    for(i=0;i<n;i++) printf("  b(%d)=%.2lf\n",i+1,a[i*(n+1)+n]);
    
    /* algorithme de Gauss-Jordan */
    cudaMalloc ((void **) &cuda_a, size);
    cudaMemcpy (cuda_a, a, size, cudaMemcpyHostToDevice);
    
    float elapsedTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventRecord(start,0);

    for(i=0; i<n; i++){
        gauss<<<n, n+1>>>(cuda_a, n, i);
    }
    
    cudaEventCreate(&stop);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start,stop);
    cudaMemcpy (a, cuda_a, size, cudaMemcpyDeviceToHost);
    
    /* Partie Affichage de la solution */

    for (int i = 0; i < n; i++){
        if(!(a[i*(n+1)+n] / a[i*(n+1)+i])){
            flag =1;
            break;
        }
    }

    if(flag != 0){
       printf("La méthode ne marche pas avec ce système :(\n");
    }else{
       printf("La solution :\n");
       for (int i = 0; i < n; i++){
        printf("x%d = %f\n", i+1, a[i*(n+1)+n] / a[i*(n+1)+i]);
    }
    }
    printf("Le temps d'exécution du programme avec le GPU est : %f seconds\n", elapsedTime/1000);
    /* desallocations */
    free(a);
    cudaFree(cuda_a);
    return 0;
}