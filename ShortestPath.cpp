
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define INFINITY  999999
#define MAX_VERTEX_NUM 200
#define FALSE 0
#define TRUE 1
#define No_City 201

typedef struct VertexType{
	char nation[100];
	char city[100];
	double latitude;
	double longitude;
	int Num_City;
}Vertex;

typedef struct ArcCell{
	char transportation[10];
	double time;
	double money;
	char *info;
}ArcCell;

typedef struct City_Order{
	int Num;
	int Order;
}City_Order;

//-------------------------------------------------
int ReturnCityNum(Vertex *v,int n,char *c){
	int i = 0;
	while(strcmp(c,v[i].city)){
		i++;
	}
	return v[i].Num_City;
}//返回城市的编号
//---------------------------------------------


void Shortest_DIJ_Time(ArcCell arcs[MAX_VERTEX_NUM][MAX_VERTEX_NUM],int v0,int vexnum,int IsShortestVia[MAX_VERTEX_NUM][MAX_VERTEX_NUM],double D[MAX_VERTEX_NUM]){
	bool final[MAX_VERTEX_NUM];
	int order = 1;
	for(int v = 0;v < vexnum;v++){
		final[v] = FALSE;
		D[v] = arcs[v0][v].time;
		for(int w = 0;w < vexnum;w++){
			IsShortestVia[v][w] = FALSE;
		}//for w
		if(D[v] < INFINITY){
			IsShortestVia[v][v0] = order;
			IsShortestVia[v][v] = ++order;
		}//if
	}//for v
	D[v0] = 0;
	final[v0] = TRUE;
	
	for(int i = 1;i<vexnum;i++){
		double min = INFINITY;
		for(int w = 0;w < vexnum;w++){
			if(!final[w]){
				if(D[w] < min){
					v = w;
					min = D[w];
				}//if D
			}//if final
		}//for w
		final[v] = TRUE;
		for(w = 0;w < vexnum;w++){
			if(!final[w]&&(min+arcs[v][w].time < D[w])){
				D[w] = min + arcs[v][w].time;
				for(int a = 0;a < vexnum;a++){
					IsShortestVia[w][a] = IsShortestVia[v][a];
					IsShortestVia[w][w] = order;
					order++;
				}//for a
			}//if
		}//for w
	}//for i
}//void Shortest_DIJ_Time

void Shortest_DIJ_Money(ArcCell arcs[MAX_VERTEX_NUM][MAX_VERTEX_NUM],int v0,int vexnum,int IsShortestVia[MAX_VERTEX_NUM][MAX_VERTEX_NUM],double D[MAX_VERTEX_NUM]){
	bool final[MAX_VERTEX_NUM];
	int order = 1;
	for(int v = 0;v < vexnum;v++){
		final[v] = FALSE;
		D[v] = arcs[v0][v].money;
		for(int w = 0;w < vexnum;w++){
			IsShortestVia[v][w] = FALSE;
		}//for w
		if(D[v] < INFINITY){
			IsShortestVia[v][v0] = order;
			IsShortestVia[v][v] = ++order;
		}//if
	}//for v
	D[v0] = 0;
	final[v0] = TRUE;
	
	for(int i = 1;i<vexnum;i++){
		double min = INFINITY;
		for(int w = 0;w < vexnum;w++){
			if(!final[w]){
				if(D[w] < min){
					v = w;
					min = D[w];
				}//if D
			}//if final
		}//for w
		final[v] = TRUE;
		for(w = 0;w < vexnum;w++){
			if(!final[w]&&(min+arcs[v][w].money < D[w])){
				D[w] = min + arcs[v][w].money;
				for(int a = 0;a < vexnum;a++){
					IsShortestVia[w][a] = IsShortestVia[v][a];
					IsShortestVia[w][w] = order;
					order++;
				}//for a
			}//if
		}//for w
	}//for i
}//void Shortest_DIJ_Money

//-------------------------------------------------------------------------------------------------------

void main(){
	FILE *fp_cities = fopen("E:\\cities.csv","rt");
	if(!fp_cities) printf("Can not open the file !\n");
	FILE *fp_routes = fopen("E:\\routes.csv","rt");
	if(!fp_routes) printf("Can not open the file !\n");
	//打开文件
	
	//--------------------------------------------------

	int Vex_Num = 0;
	int i = 0;
	int j = 0;
	char Vex_String[128];
	Vertex NatCityPosi[MAX_VERTEX_NUM];
	for( i = 0;i < MAX_VERTEX_NUM;i++){
		strcpy(NatCityPosi[i].nation,"\0");
		strcpy(NatCityPosi[i].city,"\0");
		NatCityPosi[i].latitude = 0;
		NatCityPosi[i].longitude = 0;
		NatCityPosi[i].Num_City = 0;
	}//初始化NatCityPosi
	for(i = 0;fgets(Vex_String,128,fp_cities)!=NULL;i++){
		char *temp_nation = strtok(Vex_String,",");
		strcpy(NatCityPosi[i].nation,temp_nation);
		char *temp_city = strtok(NULL,",");
		strcpy(NatCityPosi[i].city,temp_city);
		NatCityPosi[i].latitude = atof(strtok(NULL,","));
		NatCityPosi[i].longitude = atof(strtok(NULL,","));
		NatCityPosi[i].Num_City = i;
		Vex_Num++;
	}//输入顶点信息

//---------------------------------------------------------------

	ArcCell Tr_Tm_Mn_If_Matrix[MAX_VERTEX_NUM][MAX_VERTEX_NUM];
	for(i = 0;i < MAX_VERTEX_NUM;i++){
		for(j = 0;j < MAX_VERTEX_NUM;j++){
			strcpy(Tr_Tm_Mn_If_Matrix[i][j].transportation,"\0");
			Tr_Tm_Mn_If_Matrix[i][j].time = INFINITY;
			Tr_Tm_Mn_If_Matrix[i][j].money = INFINITY;
			Tr_Tm_Mn_If_Matrix[i][j].info = (char*)malloc(300*sizeof(char));
			strcpy(Tr_Tm_Mn_If_Matrix[i][j].info,"\0");
		}
	}//初始化邻接矩阵
	
	char ArcString[1000];
	while(fgets(ArcString,1000,fp_routes)!=NULL){
		char *temp_departure = strtok(ArcString,",");
 		i = ReturnCityNum(NatCityPosi,Vex_Num,temp_departure);
		char *temp_arrival = strtok(NULL,",");
		j = ReturnCityNum(NatCityPosi,Vex_Num,temp_arrival);
		strcpy(Tr_Tm_Mn_If_Matrix[i][j].transportation,strtok(NULL,","));
		Tr_Tm_Mn_If_Matrix[i][j].time = atof(strtok(NULL,","));
		Tr_Tm_Mn_If_Matrix[i][j].money = atof(strtok(NULL,","));
		strcpy(Tr_Tm_Mn_If_Matrix[i][j].info,strtok(NULL,"\0"));
	}//建立邻接矩阵


//-------------------------------------------------------------------------------

	char path_type[10];

	printf("Please input the type of the path,fastest or cheapest : ");
	scanf("%s",path_type);
	while(strcmp("fastest",path_type)&&strcmp("cheapest",path_type)){
		printf("\nYou have input a wrong type !\nPlease input the type of the path,fastest or cheapest : ");
		scanf("%s",path_type);
	}//输入路径类型


	int city_num_departure = No_City;
	int city_num_arrival = No_City;
	City_Order city_order[MAX_VERTEX_NUM] = {0};


	if(!strcmp("fastest",path_type)){
		int IsShortestVia[MAX_VERTEX_NUM][MAX_VERTEX_NUM];
		double distance[MAX_VERTEX_NUM];
		printf("Please input a departure : ");
		char city_departure[50];
		scanf("%s",city_departure);	
		city_num_departure = ReturnCityNum(NatCityPosi,Vex_Num,city_departure);
		Shortest_DIJ_Time(Tr_Tm_Mn_If_Matrix,city_num_departure,Vex_Num,IsShortestVia,distance);	
		printf("Please input a arrival : ");
		char city_arrival[50];
		scanf("%s",city_arrival);
		city_num_arrival = ReturnCityNum(NatCityPosi,Vex_Num,city_arrival);

		for(i = 0;i < Vex_Num;i++){
			city_order[i].Num = NatCityPosi[i].Num_City;
			city_order[i].Order = IsShortestVia[city_num_arrival][i];
		}

		for(i = 0;i <Vex_Num-1;i++){
			for(j = 0;j < Vex_Num;j++){
				if(city_order[i].Order < city_order[j].Order){
					City_Order temp = city_order[i];
					city_order[i] = city_order[j];
					city_order[j] = temp;
				}//if
			}// for j
		}// for i

		for(i = 0;i < Vex_Num;i++){
			if(city_order[i].Order!=0) printf("%s\n",NatCityPosi[city_order[i].Num].city);
		}
	}// if fastest
// //--------得到最快路径

	if(!strcmp("cheapest",path_type)){
		int IsShortestVia[MAX_VERTEX_NUM][MAX_VERTEX_NUM];
		double distance[MAX_VERTEX_NUM];
		printf("Please input a departure : ");
		char city_departure[50];
		scanf("%s",city_departure);
		city_num_departure = ReturnCityNum(NatCityPosi,Vex_Num,city_departure);
		Shortest_DIJ_Money(Tr_Tm_Mn_If_Matrix,city_num_departure,Vex_Num,IsShortestVia,distance);	
		printf("Please input a arrival : ");
		char city_arrival[50];
		scanf("%s",city_arrival);
		city_num_arrival = ReturnCityNum(NatCityPosi,Vex_Num,city_arrival);
		
		for(i = 0;i < Vex_Num;i++){
			city_order[i].Num = NatCityPosi[i].Num_City;
			city_order[i].Order = IsShortestVia[city_num_arrival][i];
		}
		
		for(i = 0;i <Vex_Num-1;i++){
			for(j = 0;j < Vex_Num;j++){
				if(city_order[i].Order < city_order[j].Order){
					City_Order temp = city_order[i];
					city_order[i] = city_order[j];
					city_order[j] = temp;
				}//if
			}// for j
		}// for i
		
		for(i = 0;i < Vex_Num;i++){
			if(city_order[i].Order!=0) printf("%s\n",NatCityPosi[city_order[i].Num].city);
		}
	}//if cheapset

// //----------得到最便宜路径



	char strFlie[200] = {"E:\\==sampleOutput"};
	strcat(strFlie,NatCityPosi[city_num_departure].nation);
	strcat(strFlie,NatCityPosi[city_num_arrival].nation);
	strcat(strFlie,path_type);
	strcat(strFlie,".html");
	FILE *fp_output=fopen(strFlie,"w");

	fprintf(fp_output,"<HTML><HEAD><TITLE>Shortest path from %s to %s</TITLE></HEAD><script type='text/javascript' src='http://maps.google.com/maps/api/js?sensor=false'></script><script>function initialize() { var myOptions = { zoom: 3, center: new google.maps.LatLng(0, 0), mapTypeId: google.maps.MapTypeId.ROADMAP};var map=new google.maps.Map(document.getElementById('map'), myOptions);\n\n",NatCityPosi[city_num_departure].nation,NatCityPosi[city_num_arrival].nation);

	int vexnum1 = No_City;
	int vexnum2 = No_City;
	int m = 0;
	int n = 0;

	for(i = 0;i < Vex_Num;i++){
		if(city_order[i].Order!=0){
			vexnum1 = city_order[i].Num;
			for(j = i+1;j < Vex_Num;j++){
				if(city_order[j].Order!=0){
					vexnum2 = city_order[j].Num;

					fprintf(fp_output,"var marker%d = new google.maps.Marker({ position: new google.maps.LatLng(%lf, %lf), map: map, title:'%s, %s'});\n",m,NatCityPosi[vexnum1].latitude,NatCityPosi[vexnum1].longitude,NatCityPosi[vexnum1].city,NatCityPosi[vexnum1].nation);
					fprintf(fp_output,"var marker%d = new google.maps.Marker({ position: new google.maps.LatLng(%lf, %lf), map: map, title:'%s, %s'})\n",m+1,NatCityPosi[vexnum2].latitude,NatCityPosi[vexnum2].longitude,NatCityPosi[vexnum2].city,NatCityPosi[vexnum2].nation);
					fprintf(fp_output,"var contentString%d = \"%s, %s --> %s, %s (%s - %lf hours - $%lf)\";",n,NatCityPosi[vexnum1].city,NatCityPosi[vexnum1].nation,NatCityPosi[vexnum2].city,NatCityPosi[vexnum2].nation,Tr_Tm_Mn_If_Matrix[vexnum1][vexnum2].transportation,Tr_Tm_Mn_If_Matrix[vexnum1][vexnum2].time,Tr_Tm_Mn_If_Matrix[vexnum1][vexnum2].money);
					fprintf(fp_output,"var path%d = new google.maps.Polyline({ path: [new google.maps.LatLng(%lf, %lf), new google.maps.LatLng(%lf, %lf)], strokeColor: '#0000FF', strokeOpacity: 1.0, strokeWeight: 2});",n,NatCityPosi[vexnum1].latitude,NatCityPosi[vexnum1].longitude,NatCityPosi[vexnum2].latitude,NatCityPosi[vexnum2].longitude);
					fprintf(fp_output,"path%d.setMap(map); google.maps.event.addListener(path%d, 'click', function(event) { alert(contentString%d); });\n\n",n,n,n);
					
					m = m+2;
					n++;

				}// if j
				break;
			}// for j

		}// if i

	}// for i

	
	fprintf(fp_output,"\n} google.maps.event.addDomListener(window, 'load', initialize); </script></HEAD><BODY><div id='map' style='width:100%%;height:100%%;'></div></BODY></HTML>\n");	

	//---------------------------------------------------
	
	fclose(fp_cities);
	fclose(fp_routes);
	fclose(fp_output);
}