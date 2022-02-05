#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main()
{
    int n, i, j, k, flag = 0, c;
    double **a, akk, aik;
    double end, start, time;
    printf("Entrez la dimension du système:");
    scanf("%d",&n);
    a=(double **)malloc(n*sizeof(double*));
    for(i=0;i<n;i++) a[i]=(double *)malloc((n+1)*sizeof(double));
    /* saisie des coefficients de la matrice A */
    printf("Entrez les coefficients de la matrice A, ligne par ligne :\n");
    for(i=0;i<n;i++) for(j=0;j<n;j++) {printf("Entrez le coefficient a(%d,%d)=",i+1,j+1); scanf("%lf",&a[i][j]);}
    /*a[0][0] = 0.36; a[0][1] = 0.48; a[0][2] = 195; a[0][3] = 7; a[0][4] = 0.6;      a[0][5] = 9.92;
    a[1][0] = 0.29; a[1][1] = 0.46; a[1][2] = 105; a[1][3] = 3.68; a[1][4] = 1.66;      a[1][5] = 7.95;
    a[2][0] = 0.2; a[2][1] = 0.5; a[2][2] = 138; a[2][3] = 2.94; a[2][4] = 1.91;      a[2][5] = 7.54;
    a[3][0] = 0.32; a[3][1] = 0.46; a[3][2] = 10; a[3][3] = 2.63; a[3][4] = 0.81;      a[3][5] = 7.12;
    a[4][0] = 0.16; a[4][1] = 0.42; a[4][2] = 360; a[4][3] = 0.6; a[4][4] = 0.45;      a[4][5] = 7;*/
    
    printf("Rappel des coefficients saisis:\n");
    for(i=0;i<n;i++) {for(j=0;j<n;j++) printf("  a(%d,%d)=%.2lf",i+1,j+1,a[i][j]);printf("\n");}
    /* saisie des composantes du vecteur second membre b */
    printf("Entrez les composantes du vecteur second membre :\n");
    for(i=0;i<n;i++) {printf("Entrez la composante b(%d)=",i+1); scanf("%lf",&a[i][n]);}
    printf("Rappel des composantes saisies:\n");
    for(i=0;i<n;i++) printf("  b(%d)=%.2lf\n",i+1,a[i][n]);
    
    /* algorithme de Gauss-Jordan */
    start= omp_get_wtime();
        for(i=0; i<n; i++)
        {
            #pragma omp parallel for shared(i) private(j, k)
            for (j = 0; j < n; j++) {
                // Excluding all i == j
                if (i != j) {
                    // Converting Matrix to reduced row
                    // echelon form(diagonal matrix)
                    double pro = a[j][i] / a[i][i];
                    for (k = 0; k <= n; k++)                
                        a[j][k] = a[j][k] - (a[i][k]) * pro;               
                }
            }
        }
    end = omp_get_wtime();
    time = end-start;
    /* Partie Affichage de la solution */

    for (int i = 0; i < n; i++){
        if(!(a[i][n] / a[i][i])){
            flag =1;
            break;
        }
    }

    if(flag != 0){
       printf("La méthode ne marche pas avec ce système :(\n");
    }else{
       printf("La solution :\n");
       for (int i = 0; i < n; i++){
        printf("x%d = %f\n", i+1, a[i][n] / a[i][i]);
    }
    }
    printf("Le temps d'exécution du programme avec OpenMP est : %f seconds\n", time);
    /* desallocations */
    for(i=0;i<n+1;i++) free(a[i]);
    free(a);
    return 0;
}
