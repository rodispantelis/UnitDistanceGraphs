import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Scanner;

public class Validation {
	
	//Results file
	String filename="E:\\Files\\workspace\\unit\\repository\\results\\heuristic-search-results-22K.csv"; 
	
	//Files to store graph elements
	String nodesfile="E:\\Files\\workspace\\unit\\repository\\results\\nodes.csv";
	String edgesfile="E:\\Files\\workspace\\unit\\repository\\results\\edges.csv";
	
	//Intermediate storage
	ArrayList<int[]> coords=new ArrayList<int[]>();
	ArrayList<double[]> coordsd=new ArrayList<double[]>();
	
	boolean storegraph=true; //Set true to store nodes and edges in text files
	int unities=0;
	int errors=0;
	int unities2=0;
	int errors2=0;
	int ns=0, ns2=0;
	int it=600;				//Number of iterations
	int lines=0;
	int lines2=0;
	int step=1000;			//points to process per iteration

	public static void main(String[] args) {
		Validation v = new Validation();
		v.read();
		v.read2();
		
		if(v.storegraph) { //store graph elements
			v.storenodes();
			System.out.println("graph nodes are stored in file: "+v.nodesfile);
			v.storeedges();
			System.out.println("graph edges are stored in file: "+v.edgesfile);
		}
		
		//validate unit distance pairs and produce statistics per step
		int ii=1;
		while(ii*v.step<=v.coords.size()) {
			v.it=ii;
			v.lines2=ii*v.step;
			v.lines=ii*v.step;
			v.validate();
			
			//Uncomment for floating point coordinates approximate results -  Not Recommended
			//v.validate2(); 
			
			ii++;
		}
	}
	
	//read results file integer coordinates
	public void read() {
		coords.clear();
		File pr=new File(filename);
		int cnt=0;

		try{
	    	Scanner scanner = new Scanner(pr);
	    	
	    	// Skip header
	    	if (scanner.hasNextLine()) {
	    	    scanner.nextLine();
	    	}
	    	
	    	while(scanner.hasNext() && cnt<(it*step)){
	    		String[] params= scanner.nextLine().split(",");
	    		int n[]= {Integer.parseInt(params[0]),Integer.parseInt(params[1]),Integer.parseInt(params[2]),Integer.parseInt(params[3])};
	    		coords.add(n);
	    		cnt++;
	    	}
	    	scanner.close();
		}
		catch (IOException e) {
		       e.printStackTrace();
		}
	}
	
	//read results file floating point coordinates
	public void read2() {
		coordsd.clear();
		File pr=new File(filename);
		int cnt=0;

		try{
	    	Scanner scanner = new Scanner(pr);
	    	
	    	// Skip header
	    	if (scanner.hasNextLine()) {
	    	    scanner.nextLine();
	    	}
	    	
	    	while(scanner.hasNext() && cnt<(it*step)){
	    		String[] params= scanner.nextLine().split(",");
	    		double n[]= {Double.parseDouble(params[4]),Double.parseDouble(params[5])};//,Double.parseDouble(params[2]),Double.parseDouble(params[3])};
	    		coordsd.add(n);
	    		cnt++;
	    	}
	    	scanner.close();
		}
		catch (IOException e) {
		       e.printStackTrace();
		}
	}
	
	//validation and statistics
	public void validate() {
		unities=0;
		errors=0;
		ns=0;
		
		for(int i=0; i<lines;i++){			
			for(int k=i+1; k<lines;k++) {
				int s1 = coords.get(i)[0];
				int s2 = coords.get(i)[1];
				int n1 = coords.get(i)[2];
				int n2 = coords.get(i)[3];
				
				int u1 = coords.get(k)[0];
				int u2 = coords.get(k)[1];
				int v1 = coords.get(k)[2];
				int v2 = coords.get(k)[3];
				
				int x_num = s1 * u2 - u1 * s2;
				int x_den = s2 * u2;
				int y_num = n1 * v2 - v1 * n2;
				int y_den = n2 * v2;
				
	            int lhs = (x_num * x_num) * (y_den * y_den) + (y_num * y_num) * (x_den * x_den);
	            int rhs = (x_den * x_den) * (y_den * y_den);
	            
	            if(lhs == rhs) {
	            	unities++;
	            }
	            
	            if(lhs == 0) {
	            	errors++;
	            }
	            ns++;
			}
		}
		
		System.out.println("Nodes: "+lines+", Unit Distance Edges: "+unities
				+" k: "+Math.log(unities)/Math.log(lines)
				+", Errors - duplicate points: "+errors
				);
	}
	
	//store nodes in file
	public void storenodes() {
		for(int i=0; i<coords.size();i++){
			String temp=i+","+coords.get(i)[0]+","
							+coords.get(i)[1]+","
								+coords.get(i)[2]+","
									+coords.get(i)[3];
			writefile(temp, nodesfile);
		}
	}
	
	//store edges in file in Edge List format
	public void storeedges() {
		
		for(int i=0; i<coords.size();i++){			
			for(int k=i+1; k<coords.size();k++) {
				int s1 = coords.get(i)[0];
				int s2 = coords.get(i)[1];
				int n1 = coords.get(i)[2];
				int n2 = coords.get(i)[3];
				
				int u1 = coords.get(k)[0];
				int u2 = coords.get(k)[1];
				int v1 = coords.get(k)[2];
				int v2 = coords.get(k)[3];
				
				int x_num = s1 * u2 - u1 * s2;
				int x_den = s2 * u2;
				int y_num = n1 * v2 - v1 * n2;
				int y_den = n2 * v2;
				
	            int lhs = (x_num * x_num) * (y_den * y_den) + (y_num * y_num) * (x_den * x_den);
	            int rhs = (x_den * x_den) * (y_den * y_den);
	            
	            if(lhs == rhs) {
	            	String temp=i+","+k;
	            	writefile(temp, edgesfile);
	            }
			}
		}
	}
	
	//validate and produce statistics using floating point coordinates
	//approximation results Not Recommended
	public void validate2() {
		unities2=0;
		errors2=0;
		ns2=0;

		for(int i=0; i<lines2;i++){
			for(int k=i+1; k<lines2;k++) {
				double s1 = coordsd.get(i)[0];
				double n1 = coordsd.get(i)[1];
				
				double u1 = coordsd.get(k)[0];
				double v1 = coordsd.get(k)[1];

	            
				double dis=Math.sqrt(Math.pow((s1)-(u1), 2)+Math.pow((n1)-(v1), 2));
				
	            if(dis > 0.999999 && dis < 1.000001) {
	            	unities2++;
	            }
	            
	            if(dis == 0) {
	            	errors2++;
	            }
	            ns2++;
			}
		}
		
		System.out.println("Nodes: "+lines2+", Unit Distance Edges: "+unities2
				+" k: "+Math.log(unities2)/Math.log(lines2)
				+", Errors - duplicate points: "+errors2
				);
		}
	
	//write data to text file
    public void writefile(String newln, String filename2) {

        FileWriter fw = null;
        BufferedWriter bw = null;
        PrintWriter pw = null;

		try {
			try {
				fw = new FileWriter(filename2,true);
			} catch (IOException e) {
				e.printStackTrace();
			}
			bw = new BufferedWriter(fw);
			pw = new PrintWriter(bw);
			pw.println(newln);
			pw.flush();
		}finally {
	        try {
	             pw.close();
	             bw.close();
	             fw.close();
	        } catch (IOException io) { 
	        	}
		}
	}

}
